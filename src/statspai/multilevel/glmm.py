"""
Generalized linear mixed models (GLMM) via Laplace approximation.

Supported families and links (canonical only, to keep the Laplace
approximation clean and the observed information analytic):

    * ``gaussian`` — identity link.  Mostly for cross-checks against
      ``mixed()``; for production Gaussian fits use the closed-form
      linear mixed model.
    * ``binomial`` — logit link (``sp.melogit``).  Accepts both
      0/1 Bernoulli and count/trials responses via ``trials=``.
    * ``poisson``  — log link (``sp.mepoisson``).  Optional exposure
      offsets via ``offset=``.

Model:

    η_ij = x_ij' β + z_ij' u_j,    u_j ~ N(0, G),
    y_ij | u_j ~ Family( g⁻¹(η_ij) ).

The integrated log-likelihood

    ℓ(β, θ) = Σ_j log ∫ f(y_j | β, u_j) φ(u_j; 0, G) du_j

has no closed form for non-Gaussian families; we approximate each
integral by a first-order Laplace expansion around the conditional
mode û_j, giving

    ℓ_j(β, θ) ≈ log f(y_j | β, û_j) − ½ û_j' G⁻¹ û_j
               − ½ log |G| − ½ log |Z_j' W_j Z_j + G⁻¹|,

where W_j is the diagonal GLM weight matrix at û_j.  The outer
optimisation over (β, θ) is done with L-BFGS; the inner optimisation
for û_j is a few Newton steps.

Fixed-effect covariance uses the standard GLMM formula

    Cov(β̂) = ( Σ_j X_j' W_j X_j
              − Σ_j (X_j' W_j Z_j) H_j⁻¹ (Z_j' W_j X_j) )⁻¹,

which is the observed information implied by the Laplace approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

import warnings

from ._core import (
    _GroupBlock,
    _as_str_list,
    _group_blocks,
    _initial_theta,
    _n_cov_params,
    _prepare_frame,
    _unpack_G,
)


# ---------------------------------------------------------------------------
# Exponential-family specs
# ---------------------------------------------------------------------------

class _Family:
    """Protocol for canonical-link exponential families."""

    name: str

    @staticmethod
    def inv_link(eta: np.ndarray) -> np.ndarray:          # g⁻¹(η)
        raise NotImplementedError

    @staticmethod
    def variance(mu: np.ndarray) -> np.ndarray:           # V(μ) for canonical link
        raise NotImplementedError

    @staticmethod
    def log_lik(y: np.ndarray, mu: np.ndarray, weight: np.ndarray) -> float:
        raise NotImplementedError


class _Gaussian(_Family):
    name = "gaussian"

    @staticmethod
    def inv_link(eta):
        return eta

    @staticmethod
    def variance(mu):
        return np.ones_like(mu)

    @staticmethod
    def log_lik(y, mu, weight):
        # weight carries the residual inverse-variance (set by the caller).
        return float(-0.5 * np.sum(weight * (y - mu) ** 2))


class _Binomial(_Family):
    name = "binomial"

    @staticmethod
    def inv_link(eta):
        # Numerically stable logistic.
        out = np.empty_like(eta)
        pos = eta >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
        ex = np.exp(eta[~pos])
        out[~pos] = ex / (1.0 + ex)
        return out

    @staticmethod
    def variance(mu):
        return mu * (1.0 - mu)

    @staticmethod
    def log_lik(y, mu, weight):
        # y in counts, weight in trials.
        mu = np.clip(mu, 1e-12, 1 - 1e-12)
        return float(np.sum(y * np.log(mu) + (weight - y) * np.log(1.0 - mu)))


class _Poisson(_Family):
    name = "poisson"

    @staticmethod
    def inv_link(eta):
        return np.exp(np.clip(eta, -30.0, 30.0))

    @staticmethod
    def variance(mu):
        return mu

    @staticmethod
    def log_lik(y, mu, weight):
        # weight ≡ 1 for the standard Poisson likelihood.  We drop the
        # log(y!) constant; it is immaterial to optimisation and LRT.
        mu = np.clip(mu, 1e-300, None)
        return float(np.sum(y * np.log(mu) - mu))


_FAMILIES: Dict[str, _Family] = {
    "gaussian": _Gaussian(),
    "binomial": _Binomial(),
    "poisson": _Poisson(),
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MEGLMResult:
    """Container for GLMM (``meglm``, ``melogit``, ``mepoisson``) fits."""

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
    def n_params(self) -> int:
        return self.n_fixed + self._n_cov_params

    @property
    def aic(self) -> float:
        return 2.0 * self.n_params - 2.0 * self.log_likelihood

    @property
    def bic(self) -> float:
        return self.n_params * np.log(self.n_obs) - 2.0 * self.log_likelihood

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        z = stats.norm.ppf(1 - alpha / 2)
        lo = self.fixed_effects - z * self._se_fixed
        hi = self.fixed_effects + z * self._se_fixed
        return pd.DataFrame({"lower": lo, "upper": hi})

    def odds_ratios(self) -> pd.DataFrame:
        """Exponentiated fixed effects and their Wald CIs (logit models)."""
        if self.family != "binomial":
            raise ValueError("odds_ratios() only meaningful for binomial GLMMs")
        ci = self.conf_int()
        return pd.DataFrame({
            "OR": np.exp(self.fixed_effects),
            "lower": np.exp(ci["lower"]),
            "upper": np.exp(ci["upper"]),
        })

    def incidence_rate_ratios(self) -> pd.DataFrame:
        """Exponentiated coefficients for Poisson GLMMs."""
        if self.family != "poisson":
            raise ValueError("IRR only meaningful for Poisson GLMMs")
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
    # LaTeX / plot — round out the unified result contract
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
        fam = _FAMILIES[self.family]
        return pd.Series(fam.inv_link(eta), index=data.index, name="mu")


# ---------------------------------------------------------------------------
# Inner mode-finder (Newton on u_j given β, G)
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

        if family.name == "binomial":
            W = weights * mu * (1.0 - mu)
            grad_data = block.Z.T @ (block.y - weights * mu)
        elif family.name == "poisson":
            W = mu
            grad_data = block.Z.T @ (block.y - mu)
        elif family.name == "gaussian":
            W = weights
            grad_data = block.Z.T @ (weights * (block.y - mu))
        else:  # pragma: no cover
            raise NotImplementedError(family.name)

        grad = grad_data - Ginv @ u
        H = block.Z.T @ (W[:, None] * block.Z) + Ginv

        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break

        # Simple step damping: keep ‖step‖ ≤ 5·‖u‖ to avoid blow-up
        # when the mode is very far from the current iterate.
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

    # Final H and log|H|.
    eta = block.X @ beta + block.Z @ u + offset
    mu = family.inv_link(eta)
    if family.name == "binomial":
        W = weights * mu * (1.0 - mu)
    elif family.name == "poisson":
        W = mu
    else:
        W = weights
    H = block.Z.T @ (W[:, None] * block.Z) + Ginv
    sign, logdet_H = np.linalg.slogdet(H)
    if sign <= 0:
        logdet_H = np.inf
        converged = False
    return u, H, logdet_H, converged


# ---------------------------------------------------------------------------
# Objective: negative Laplace log-likelihood
# ---------------------------------------------------------------------------


def _laplace_nll(
    theta: np.ndarray,
    blocks: List[_GroupBlock],
    weights_list: List[np.ndarray],
    offsets_list: List[np.ndarray],
    p_fixed: int,
    q_random: int,
    cov_type: str,
    family: _Family,
    u_cache: List[np.ndarray],
):
    """Packed theta = [β ; cov_params]."""
    beta = theta[:p_fixed]
    cov_params = theta[p_fixed:]
    G = _unpack_G(cov_params, q_random, cov_type)
    try:
        sign, logdet_G = np.linalg.slogdet(G)
        if sign <= 0:
            return 1e12
        Ginv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        return 1e12

    nll = 0.0
    for j, block in enumerate(blocks):
        w = weights_list[j]
        off = offsets_list[j]
        u_hat, H_j, logdet_H, _inner_converged = _find_mode(
            block,
            beta,
            G,
            Ginv,
            family,
            w,
            off,
            u_cache[j],
        )
        u_cache[j] = u_hat  # warm-start for the next evaluation

        eta = block.X @ beta + block.Z @ u_hat + off
        mu = family.inv_link(eta)
        ll_data = family.log_lik(block.y, mu, w)
        quad = float(u_hat @ Ginv @ u_hat)

        # Laplace log-integrand (dropping terms independent of θ):
        #   log L_j ≈ ll_data − 0.5 q log(2π) − 0.5 log|G| − 0.5 u' G⁻¹ u
        #            + 0.5 q log(2π) − 0.5 log|H_j|
        #          = ll_data − 0.5 log|G| − 0.5 u' G⁻¹ u − 0.5 log|H_j|
        nll -= ll_data - 0.5 * logdet_G - 0.5 * quad - 0.5 * logdet_H

    return nll


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
    maxiter: int = 300,
    tol: float = 1e-6,
    alpha: float = 0.05,
) -> MEGLMResult:
    """
    Fit a generalised linear mixed model by Laplace approximation.

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
        ``'gaussian'``, ``'binomial'`` or ``'poisson'``.
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
    maxiter, tol, alpha
        Optimisation controls / CI width.

    Returns
    -------
    MEGLMResult
    """
    fam_key = family.lower()
    if fam_key not in _FAMILIES:
        raise ValueError(
            f"family must be one of {list(_FAMILIES)}, got {family!r}"
        )
    fam = _FAMILIES[fam_key]
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

    x_fixed = list(x_fixed)
    x_random_cols: List[str] = list(x_random) if x_random is not None else []
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
        elif fam_key == "binomial":
            w = np.ones(block.n)
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

    # Starting values for β: run a GLM (no random effects) via IRLS.
    X_all = np.vstack([b.X for b in blocks])
    y_all = np.concatenate([b.y for b in blocks])
    w_all = np.concatenate(weights_list)
    off_all = np.concatenate(offsets_list)
    beta0 = _irls_init(X_all, y_all, w_all, off_all, fam)

    theta_cov0 = _initial_theta(q_random, cov_type, s2_init=0.3)
    theta0 = np.concatenate([beta0, theta_cov0])

    u_cache = [np.zeros(q_random) for _ in blocks]

    res = minimize(
        _laplace_nll,
        theta0,
        args=(blocks, weights_list, offsets_list, p_fixed, q_random, cov_type, fam, u_cache),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": tol, "gtol": tol},
    )
    outer_converged = bool(res.success)

    beta_hat = res.x[:p_fixed]
    G_hat = _unpack_G(res.x[p_fixed:], q_random, cov_type)
    Ginv = np.linalg.inv(G_hat)

    # BLUPs and fixed-effect info matrix via mode-finding at the optimum.
    blup_rows: List[Dict[str, float]] = []
    blup_dict: Dict[Any, np.ndarray] = {}
    keys = []

    info = np.zeros((p_fixed, p_fixed))
    inner_failures = 0
    for j, (block, w, off) in enumerate(zip(blocks, weights_list, offsets_list)):
        u0 = u_cache[j]  # warm-started cache from the last NLL evaluation
        u_hat, H_j, _, inner_ok = _find_mode(
            block, beta_hat, G_hat, Ginv, fam, w, off, u0
        )
        if not inner_ok:
            inner_failures += 1
        eta = block.X @ beta_hat + block.Z @ u_hat + off
        mu = fam.inv_link(eta)
        if fam.name == "binomial":
            W = w * mu * (1.0 - mu)
        elif fam.name == "poisson":
            W = mu
        else:
            W = w
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
            f"GLMM Laplace inner Newton failed to converge for "
            f"{inner_failures}/{len(blocks)} clusters; standard errors "
            "and log-likelihood may be unreliable for those groups.",
            RuntimeWarning,
            stacklevel=2,
        )

    random_effects_df = pd.DataFrame(blup_rows, index=keys)
    random_effects_df.index.name = group

    # Variance components
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

    link = {"gaussian": "identity", "binomial": "logit", "poisson": "log"}[fam_key]

    return MEGLMResult(
        fixed_effects=pd.Series(beta_hat, index=fixed_names),
        random_effects=random_effects_df,
        variance_components=vc,
        blups=blup_dict,
        n_obs=int(np.sum(w_all) if fam_key == "binomial" and trials else len(df)),
        n_groups=len(blocks),
        log_likelihood=float(-res.fun),
        family=fam_key,
        link=link,
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
        _method="laplace",
        _cov_type=cov_type,
        _alpha=alpha,
        _n_cov_params=n_cov_pars,
        _offset_name=offset,
    )


def melogit(
    data: pd.DataFrame,
    y: str,
    x_fixed: Sequence[str],
    group: str,
    x_random: Optional[Sequence[str]] = None,
    trials: Optional[str] = None,
    **kw: Any,
) -> MEGLMResult:
    """Random-effects logistic regression (Stata ``melogit``)."""
    return meglm(
        data, y, x_fixed, group,
        family="binomial",
        x_random=x_random,
        trials=trials,
        **kw,
    )


def mepoisson(
    data: pd.DataFrame,
    y: str,
    x_fixed: Sequence[str],
    group: str,
    x_random: Optional[Sequence[str]] = None,
    offset: Optional[str] = None,
    **kw: Any,
) -> MEGLMResult:
    """Random-effects Poisson regression (Stata ``mepoisson``)."""
    return meglm(
        data, y, x_fixed, group,
        family="poisson",
        x_random=x_random,
        offset=offset,
        **kw,
    )


# ---------------------------------------------------------------------------
# GLM-only IRLS to get warm-start β
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
    """Plain GLM via Fisher scoring — used only to warm-start β."""
    p = X.shape[1]
    beta = np.zeros(p)
    if fam.name == "binomial":
        # Start mu at the overall success rate.
        pbar = np.clip(np.sum(y) / max(np.sum(w), 1.0), 1e-3, 1 - 1e-3)
        beta[0] = np.log(pbar / (1.0 - pbar))
    elif fam.name == "poisson":
        beta[0] = np.log(max(np.mean(y), 1e-6))

    for _ in range(maxiter):
        eta = X @ beta + off
        mu = fam.inv_link(eta)
        if fam.name == "binomial":
            W = w * mu * (1.0 - mu)
            grad = X.T @ (y - w * mu)
        elif fam.name == "poisson":
            W = mu
            grad = X.T @ (y - mu)
        else:
            W = w
            grad = X.T @ (w * (y - mu))
        H = X.T @ (W[:, None] * X) + 1e-8 * np.eye(p)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta += step
        if np.linalg.norm(step) < tol * (1 + np.linalg.norm(beta)):
            break
    return beta


__all__ = [
    "meglm",
    "melogit",
    "mepoisson",
    "MEGLMResult",
]
