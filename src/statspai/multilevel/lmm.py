"""
Linear mixed-effects (hierarchical linear) models.

Model:

    y_ij = X_ij' β + Z_ij' u_j + ε_ij,
    u_j  ~ N(0, G),   ε_ij ~ N(0, σ² I),

with up to two **nested** grouping levels — passing a single group
name fits the classical two-level model, while passing a list such as
``group=["school", "class"]`` fits a three-level nested LMM by first
collapsing school×class into an effective innermost cluster (the
between-school variance is kept separate through a second random
intercept block).

Estimation is by profiled (RE)ML with analytical GLS for the fixed
effects and numerical optimisation over the covariance parameters.
The covariance structure of G is user-selectable:

    * ``identity``     — σ² I_q
    * ``diagonal``     — default in earlier StatsPAI versions
    * ``unstructured`` — full q×q PSD matrix (Stata ``cov(unstructured)``,
                         R ``lme4`` default).

``unstructured`` is now the default because diagonal G is rarely
realistic once a random slope is included — the intercept-slope
correlation is a first-class effect of interest.

Compared with the previous pared-down implementation, this file adds:

    * Full unstructured G via Cholesky parameterisation.
    * BLUP posterior variance (``ranef_se``) for caterpillar plots.
    * ``predict()`` with population and group-conditional options.
    * Nakagawa-Schielzeth marginal / conditional R² (``r_squared()``).
    * AIC / BIC, degree-of-freedom accounting.
    * Wald tests, LaTeX / Markdown / HTML export.
    * Three-level nested support through ``group=[outer, inner]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ..core.results import EconometricResults
from ._core import (
    _GroupBlock,
    _as_str_list,
    _group_blocks,
    _initial_theta,
    _n_cov_params,
    _prepare_frame,
    _solve_V,
    _unpack_G,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MixedResult:
    """
    Container for ``sp.mixed()`` estimation results.

    The public attributes mirror Stata ``mixed`` and R ``lme4::lmer``
    output; the underscored fields carry implementation state used by
    methods like ``predict()`` and ``r_squared()``.
    """

    fixed_effects: pd.Series
    random_effects: pd.DataFrame          # BLUPs, one row per innermost group
    variance_components: Dict[str, float]
    blups: Dict[Any, np.ndarray]
    n_obs: int
    n_groups: int
    icc: float
    log_likelihood: float

    # internal bookkeeping --------------------------------------------------
    _se_fixed: pd.Series = field(default=None, repr=False)
    _cov_fixed: np.ndarray = field(default=None, repr=False)
    _G: np.ndarray = field(default=None, repr=False)
    _sigma2: float = field(default=np.nan, repr=False)
    _blocks: List[_GroupBlock] = field(default_factory=list, repr=False)
    _group_cols: List[str] = field(default_factory=list, repr=False)
    _fixed_names: List[str] = field(default_factory=list, repr=False)
    _random_names: List[str] = field(default_factory=list, repr=False)
    _x_fixed: List[str] = field(default_factory=list, repr=False)
    _x_random: List[str] = field(default_factory=list, repr=False)
    _y_name: str = field(default="", repr=False)
    _method: str = field(default="reml", repr=False)
    _cov_type: str = field(default="unstructured", repr=False)
    _converged: bool = field(default=True, repr=False)
    _alpha: float = field(default=0.05, repr=False)
    _n_cov_params: int = field(default=0, repr=False)
    _lr_test: Optional[Dict[str, float]] = field(default=None, repr=False)
    _ranef_se: Optional[pd.DataFrame] = field(default=None, repr=False)
    _nakagawa: Optional[Dict[str, float]] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def params(self) -> pd.Series:
        """Alias used by the unified StatsPAI result contract."""
        return self.fixed_effects

    @property
    def bse(self) -> pd.Series:
        """Fixed-effect standard errors (Stata style alias)."""
        return self._se_fixed

    @property
    def std_errors(self) -> pd.Series:
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
        """Total free parameters (fixed + variance components + sigma²)."""
        return self.n_fixed + self._n_cov_params + 1

    @property
    def aic(self) -> float:
        return 2.0 * self.n_params - 2.0 * self.log_likelihood

    @property
    def bic(self) -> float:
        return self.n_params * np.log(self.n_obs) - 2.0 * self.log_likelihood

    def conf_int(self, alpha: Optional[float] = None) -> pd.DataFrame:
        """Wald confidence intervals for the fixed effects."""
        alpha = self._alpha if alpha is None else alpha
        z = stats.norm.ppf(1 - alpha / 2)
        lo = self.fixed_effects - z * self._se_fixed
        hi = self.fixed_effects + z * self._se_fixed
        return pd.DataFrame({"lower": lo, "upper": hi})

    def ranef(self, conditional_se: bool = False) -> pd.DataFrame:
        """
        Return BLUPs (posterior means of the random effects).

        Parameters
        ----------
        conditional_se
            If ``True``, return a second DataFrame of posterior standard
            errors computed from the conditional variance
            ``Var(u_j | y) = G − G Z_j' V_j^{−1} Z_j G``.
        """
        if conditional_se:
            if self._ranef_se is None:
                raise RuntimeError("conditional SEs were not stored with this result")
            return self.random_effects.copy(), self._ranef_se.copy()
        return self.random_effects.copy()

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        data: Optional[pd.DataFrame] = None,
        include_random: bool = True,
    ) -> pd.Series:
        """
        Fitted / predicted values.

        ``include_random=True`` returns X β̂ + Z û (conditional on the
        realised random effects), which is the Stata ``predict , fitted``
        default.  ``include_random=False`` returns the marginal /
        population prediction X β̂.

        For unseen groups (i.e. groups absent from training data), the
        random-effect contribution is treated as zero, which is the
        best-linear unbiased prediction when nothing is known about the
        new cluster.
        """
        if data is None:
            # Use the training blocks.
            out = np.empty(self.n_obs)
            idx_start = 0
            for block in self._blocks:
                mu = block.X @ self.fixed_effects.values
                if include_random:
                    u = self.blups[block.key]
                    mu = mu + block.Z @ u
                out[idx_start : idx_start + block.n] = mu
                idx_start += block.n
            return pd.Series(out, name="yhat")

        # Arbitrary new data ------------------------------------------------
        for c in self._x_fixed + self._group_cols:
            if c not in data.columns:
                raise KeyError(f"predict(): missing column {c!r}")

        X_new = np.column_stack(
            [np.ones(len(data))] + [data[c].to_numpy(dtype=float) for c in self._x_fixed]
        )
        mu = X_new @ self.fixed_effects.values

        if include_random and self._x_random is not None:
            Z_new = np.column_stack(
                [np.ones(len(data))]
                + [data[c].to_numpy(dtype=float) for c in self._x_random]
            )
            # Map each row to its BLUP (or zero for unseen groups).
            group_key_col = _compose_group_key(data, self._group_cols)
            u_mat = np.zeros_like(Z_new)
            for i, key in enumerate(group_key_col):
                u = self.blups.get(key)
                if u is not None:
                    u_mat[i, :] = u
            mu = mu + np.einsum("ij,ij->i", Z_new, u_mat)

        return pd.Series(mu, index=data.index, name="yhat")

    # ------------------------------------------------------------------
    # R²
    # ------------------------------------------------------------------

    def r_squared(self) -> Dict[str, float]:
        """
        Nakagawa & Schielzeth (2013) marginal and conditional R².

        Definitions, in our Gaussian LMM:

            σ²_f  = variance of the fixed-effect predictions Xβ
            σ²_r  = sum of random-effect variance contributions
                   (trace of G weighted by Z'Z/n for random slopes)
            σ²_ε  = residual variance

            R²_marginal    = σ²_f / (σ²_f + σ²_r + σ²_ε)
            R²_conditional = (σ²_f + σ²_r) / (σ²_f + σ²_r + σ²_ε)
        """
        if self._nakagawa is not None:
            return dict(self._nakagawa)

        # σ²_f
        fitted_fixed = np.concatenate(
            [block.X @ self.fixed_effects.values for block in self._blocks]
        )
        sigma2_f = float(np.var(fitted_fixed, ddof=0))

        # σ²_r : trace(G · E[Z'Z/n])  (population averaged over observations)
        if self._G is not None and len(self._blocks) > 0:
            ZtZ_sum = np.zeros_like(self._G)
            n_total = 0
            for b in self._blocks:
                ZtZ_sum += b.Z.T @ b.Z
                n_total += b.n
            mean_ZtZ = ZtZ_sum / n_total
            sigma2_r = float(np.trace(self._G @ mean_ZtZ))
        else:
            sigma2_r = 0.0

        sigma2_e = float(self._sigma2)
        denom = sigma2_f + sigma2_r + sigma2_e
        r2_marg = sigma2_f / denom if denom > 0 else np.nan
        r2_cond = (sigma2_f + sigma2_r) / denom if denom > 0 else np.nan

        out = {
            "marginal": r2_marg,
            "conditional": r2_cond,
            "var_fixed": sigma2_f,
            "var_random": sigma2_r,
            "var_residual": sigma2_e,
        }
        self._nakagawa = out
        return dict(out)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def wald_test(self, restrictions: "Sequence[str] | np.ndarray | pd.DataFrame") -> Dict[str, float]:
        """
        Wald joint test of linear restrictions on the fixed effects.

        ``restrictions`` can be:

        * A list of parameter names — tests those coefficients are zero.
        * A 2-D array / DataFrame R with ``R β = 0`` (rows = restrictions,
          columns ordered as ``self.fixed_effects.index``).
        """
        beta = self.fixed_effects.values
        cov = self._cov_fixed
        names = list(self.fixed_effects.index)

        if isinstance(restrictions, (list, tuple)) and all(
            isinstance(r, str) for r in restrictions
        ):
            idx = [names.index(r) for r in restrictions]
            R = np.zeros((len(idx), len(beta)))
            for row, col in enumerate(idx):
                R[row, col] = 1.0
        elif isinstance(restrictions, pd.DataFrame):
            R = restrictions.reindex(columns=names).fillna(0.0).to_numpy()
        else:
            R = np.asarray(restrictions, dtype=float)

        if R.ndim == 1:
            R = R[None, :]
        Rbeta = R @ beta
        middle = R @ cov @ R.T
        try:
            chi2 = float(Rbeta @ np.linalg.solve(middle, Rbeta))
        except np.linalg.LinAlgError:
            chi2 = np.nan
        df = R.shape[0]
        p = float(1.0 - stats.chi2.cdf(chi2, df)) if chi2 == chi2 else np.nan
        return {"chi2": chi2, "df": df, "p_value": p}

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Stata-style formatted summary."""
        lines: List[str] = []
        w = 76
        lines.append("=" * w)
        lines.append("Mixed-effects model".center(w))
        lines.append("=" * w)

        lines.append(f"  Method:          {self._method.upper()}")
        lines.append(f"  Cov(random):     {self._cov_type}")
        lines.append(f"  No. obs:         {self.n_obs}")
        lines.append(f"  No. groups:      {self.n_groups}")
        lines.append(f"  Log-likelihood:  {self.log_likelihood:.4f}")
        lines.append(f"  AIC / BIC:       {self.aic:.3f}  /  {self.bic:.3f}")
        lines.append(f"  ICC (intercept): {self.icc:.4f}")
        lines.append(f"  Converged:       {self._converged}")
        lines.append("-" * w)

        # Fixed effects table
        z_crit = stats.norm.ppf(1 - self._alpha / 2)
        lines.append("Fixed effects:")
        hdr = f"{'':>18s} {'Coef':>10s} {'Std.Err':>10s} {'z':>8s} {'P>|z|':>8s}  [{100*(1-self._alpha):.0f}% CI]"
        lines.append(hdr)
        lines.append("-" * w)
        for var in self.fixed_effects.index:
            b = self.fixed_effects[var]
            se = self._se_fixed[var] if self._se_fixed is not None else np.nan
            z = b / se if se and se > 0 else np.nan
            p = 2 * (1 - stats.norm.cdf(abs(z))) if z == z else np.nan
            lo, hi = b - z_crit * se, b + z_crit * se
            lines.append(
                f"{var:>18s} {b:10.4f} {se:10.4f} {z:8.3f} {p:8.4f}  [{lo:8.4f}, {hi:8.4f}]"
            )

        lines.append("-" * w)
        lines.append("Variance components:")
        for name, val in self.variance_components.items():
            lines.append(f"  {name:24s}  {val:.6f}")

        # Nakagawa R² (lazy compute if blocks retained)
        if len(self._blocks) > 0:
            try:
                r2 = self.r_squared()
                lines.append("-" * w)
                lines.append(
                    f"R² (Nakagawa-Schielzeth):  marginal = {r2['marginal']:.4f}, "
                    f"conditional = {r2['conditional']:.4f}"
                )
            except Exception:
                pass

        if self._lr_test is not None:
            lines.append("-" * w)
            lr = self._lr_test
            lines.append(
                f"LR test vs. pooled OLS:  chi2({lr['df']:.0f}) = {lr['chi2']:.4f}, "
                f"Prob > chi2 = {lr['p']:.4f}"
            )
        lines.append("=" * w)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        r2 = self.r_squared() if len(self._blocks) > 0 else {"marginal": np.nan, "conditional": np.nan}
        out = ["# Linear Mixed Model\n"]
        out.append(
            f"**Method:** {self._method.upper()}   "
            f"**Cov(random):** {self._cov_type}   "
            f"**N = {self.n_obs}, Groups = {self.n_groups}**\n"
        )
        out.append("## Fixed effects\n")
        out.append("| Variable | Coef. | Std. Err. | z | P>|z| |")
        out.append("|----------|------:|----------:|---:|-----:|")
        for var in self.fixed_effects.index:
            b = self.fixed_effects[var]
            se = self._se_fixed[var]
            z = b / se if se else float("nan")
            p = 2 * (1 - stats.norm.cdf(abs(z))) if z == z else float("nan")
            out.append(f"| {var} | {b:.4f} | {se:.4f} | {z:.3f} | {p:.4f} |")
        out.append("\n## Variance components\n")
        out.append("| Component | Estimate |")
        out.append("|-----------|---------:|")
        for name, val in self.variance_components.items():
            out.append(f"| {name} | {val:.6f} |")
        out.append(
            f"\n**R² (marginal):** {r2['marginal']:.4f}  "
            f"**R² (conditional):** {r2['conditional']:.4f}  "
            f"**ICC:** {self.icc:.4f}  "
            f"**AIC:** {self.aic:.2f}  **BIC:** {self.bic:.2f}\n"
        )
        return "\n".join(out)

    def to_latex(self) -> str:
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Linear Mixed Model}",
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
            lines.append(
                f"{var} & {b:.4f} & {se:.4f} & {z:.3f} & {p:.4f} \\\\"
            )
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{5}{l}{\textit{Variance components}} \\")
        for name, val in self.variance_components.items():
            safe = name.replace("_", r"\_")
            lines.append(f"{safe} & \\multicolumn{{4}}{{r}}{{{val:.6f}}} \\\\")
        lines.append(r"\bottomrule")
        lines.append(
            rf"\multicolumn{{5}}{{l}}{{\footnotesize $N={self.n_obs}$, groups $={self.n_groups}$, "
            rf"LogL $={self.log_likelihood:.3f}$, AIC $={self.aic:.2f}$.}} \\"
        )
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        r2 = self.r_squared() if len(self._blocks) > 0 else {"marginal": float("nan"), "conditional": float("nan")}
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
            f"<h4>Linear Mixed Model &mdash; {self._method.upper()} "
            f"(cov={self._cov_type})</h4>"
            f"<p>N = {self.n_obs}, groups = {self.n_groups}, "
            f"ICC = {self.icc:.4f}, "
            f"R² marginal = {r2['marginal']:.3f}, "
            f"R² conditional = {r2['conditional']:.3f}, "
            f"AIC = {self.aic:.2f}, BIC = {self.bic:.2f}</p>"
            "<table><thead><tr><th>Variable</th><th>Coef</th><th>SE</th><th>z</th></tr></thead>"
            f"<tbody>{rows_fixed}</tbody></table>"
            "<table><thead><tr><th>Variance component</th><th>Estimate</th></tr></thead>"
            f"<tbody>{rows_vc}</tbody></table>"
            "</div>"
        )

    def cite(self) -> str:
        return (
            "@book{mcculloch2008,\n"
            "  title   = {Generalized, Linear, and Mixed Models},\n"
            "  author  = {McCulloch, Charles E. and Searle, Shayle R. and Neuhaus, John M.},\n"
            "  edition = {2nd},\n"
            "  publisher = {Wiley},\n"
            "  year    = {2008}\n"
            "}\n"
            "@article{nakagawa2013,\n"
            "  author  = {Nakagawa, Shinichi and Schielzeth, Holger},\n"
            "  title   = {A general and simple method for obtaining R² from"
            " generalized linear mixed-effects models},\n"
            "  journal = {Methods in Ecology and Evolution},\n"
            "  volume  = {4},\n"
            "  year    = {2013},\n"
            "  pages   = {133--142}\n"
            "}\n"
        )

    def to_econometric_results(self) -> EconometricResults:
        params = self.fixed_effects
        se = self._se_fixed if self._se_fixed is not None else pd.Series(
            np.nan, index=params.index
        )
        model_info = {
            "model_type": "Mixed-effects LMM",
            "method": self._method,
            "cov_type": self._cov_type,
            "converged": self._converged,
        }
        data_info = {
            "n_obs": self.n_obs,
            "n_groups": self.n_groups,
            "df_resid": self.n_obs - len(params),
            "dependent_var": self._y_name,
        }
        diagnostics = {
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "icc": self.icc,
            "variance_components": self.variance_components,
        }
        if self._lr_test is not None:
            diagnostics["lr_test"] = self._lr_test
        return EconometricResults(
            params=params,
            std_errors=se,
            model_info=model_info,
            data_info=data_info,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(
        self,
        kind: str = "caterpillar",
        variable: Optional[str] = None,
        **kwargs,
    ):
        """
        Quick diagnostic plots.

        ``kind='caterpillar'`` produces a forest plot of BLUPs with
        posterior-SE intervals (the lme4 standard) for a chosen random
        effect (default: intercept).  ``kind='residuals'`` plots the
        conditional residuals against fitted values.
        """
        import matplotlib.pyplot as plt  # local import: keep plotting optional

        if kind == "caterpillar":
            name = variable if variable is not None else self._random_names[0]
            if name not in self.random_effects.columns:
                raise ValueError(f"random effect {name!r} not in model")
            u = self.random_effects[name].copy().sort_values()
            se = (
                self._ranef_se[name].loc[u.index]
                if self._ranef_se is not None
                else pd.Series(np.nan, index=u.index)
            )
            lo = u - 1.96 * se
            hi = u + 1.96 * se

            fig, ax = plt.subplots(**{"figsize": (6, 0.2 * len(u) + 1), **kwargs})
            y_pos = np.arange(len(u))
            ax.errorbar(u.values, y_pos, xerr=1.96 * se.values, fmt="o", ms=3, lw=1)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_yticks(y_pos)
            ax.set_yticklabels([str(i) for i in u.index], fontsize=7)
            ax.set_xlabel(f"BLUP of {name}")
            ax.set_title(f"Caterpillar plot: random {name}")
            fig.tight_layout()
            return fig, ax

        if kind == "residuals":
            yhat = self.predict().values
            y_all = np.concatenate([b.y for b in self._blocks])
            resid = y_all - yhat
            fig, ax = plt.subplots(**{"figsize": (5, 4), **kwargs})
            ax.scatter(yhat, resid, s=10, alpha=0.6)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xlabel("Fitted values")
            ax.set_ylabel("Conditional residual")
            ax.set_title("Residuals vs fitted")
            fig.tight_layout()
            return fig, ax

        raise ValueError(f"unknown plot kind {kind!r}")


# ---------------------------------------------------------------------------
# Group-key composition (handles nested levels)
# ---------------------------------------------------------------------------


def _compose_group_key(data: pd.DataFrame, group_cols: Sequence[str]):
    """Return an iterable of hashable keys identical to the one used at fit."""
    if len(group_cols) == 1:
        return list(data[group_cols[0]].values)
    return [tuple(r) for r in data[list(group_cols)].itertuples(index=False, name=None)]


# ---------------------------------------------------------------------------
# Three-level nested random-intercept fit
# ---------------------------------------------------------------------------


def _three_level_nll(
    theta: np.ndarray,
    blocks_outer,
    p_fixed: int,
    n_total: int,
    reml: bool,
) -> float:
    """
    Negative (RE)ML for a 3-level nested random-intercept model.

    theta layout = [log σ²_outer, log σ²_inner, log σ²_resid].
    Blocks are outer-level (school-level) clusters — each carries the
    within-block assignment of observations to inner (class-level)
    subclusters.
    """
    sigma2_s = float(np.exp(theta[0]))
    sigma2_c = float(np.exp(theta[1]))
    sigma2_e = float(np.exp(theta[2]))

    XtVinvX = np.zeros((p_fixed, p_fixed))
    XtVinvy = np.zeros(p_fixed)
    logdet_sum = 0.0

    for y_s, X_s, inner_ids, inner_unique in blocks_outer:
        n_s = len(y_s)
        V = sigma2_e * np.eye(n_s) + sigma2_s * np.ones((n_s, n_s))
        # Add block-diagonal contribution from inner clusters.
        for c in inner_unique:
            mask = inner_ids == c
            V[np.ix_(mask, mask)] += sigma2_c
        try:
            L = np.linalg.cholesky(V)
        except np.linalg.LinAlgError:
            return 1e12
        logdet_sum += 2.0 * np.sum(np.log(np.diag(L)))
        z = np.linalg.solve(L, X_s)
        VinvX = np.linalg.solve(L.T, z)
        z = np.linalg.solve(L, y_s)
        Vinvy = np.linalg.solve(L.T, z)
        XtVinvX += X_s.T @ VinvX
        XtVinvy += X_s.T @ Vinvy

    try:
        beta = np.linalg.solve(XtVinvX, XtVinvy)
    except np.linalg.LinAlgError:
        return 1e12

    quad = 0.0
    for y_s, X_s, inner_ids, inner_unique in blocks_outer:
        n_s = len(y_s)
        V = sigma2_e * np.eye(n_s) + sigma2_s * np.ones((n_s, n_s))
        for c in inner_unique:
            mask = inner_ids == c
            V[np.ix_(mask, mask)] += sigma2_c
        r = y_s - X_s @ beta
        try:
            L = np.linalg.cholesky(V)
        except np.linalg.LinAlgError:
            return 1e12
        z = np.linalg.solve(L, r)
        quad += float(z @ z)

    nll = 0.5 * (logdet_sum + quad + n_total * np.log(2 * np.pi))
    if reml:
        sign, logdet_xtvinvx = np.linalg.slogdet(XtVinvX)
        if sign <= 0:
            return 1e12
        nll += 0.5 * logdet_xtvinvx
        nll -= 0.5 * p_fixed * np.log(2 * np.pi)
    return nll


def _fit_three_level_intercept(
    df: pd.DataFrame,
    *,
    y: str,
    x_fixed: Sequence[str],
    outer_col: str,
    inner_col: str,
    method: str,
    maxiter: int,
    tol: float,
    alpha: float,
) -> MixedResult:
    """Fit a school > class > student nested random-intercept LMM."""
    fixed_names = ["_cons"] + list(x_fixed)
    random_names = ["_cons"]
    p_fixed = len(fixed_names)
    reml = method == "reml"

    # Build outer blocks.
    blocks_outer = []
    for key, sub in df.groupby(outer_col, sort=False):
        y_s = sub[y].to_numpy(dtype=float)
        X_s = sub[["__intercept__"] + list(x_fixed)].to_numpy(dtype=float)
        inner_ids = sub[inner_col].to_numpy()
        inner_unique = np.unique(inner_ids)
        blocks_outer.append((y_s, X_s, inner_ids, inner_unique))

    n_obs = len(df)
    # Starting values from OLS
    X_all = np.vstack([b[1] for b in blocks_outer])
    y_all = np.concatenate([b[0] for b in blocks_outer])
    ols_beta, *_ = np.linalg.lstsq(X_all, y_all, rcond=None)
    resid = y_all - X_all @ ols_beta
    s2_ols = float(np.var(resid, ddof=p_fixed))
    theta0 = np.array([np.log(0.1 * s2_ols), np.log(0.1 * s2_ols), np.log(0.8 * s2_ols)])

    res = minimize(
        _three_level_nll,
        theta0,
        args=(blocks_outer, p_fixed, n_obs, reml),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": tol, "gtol": tol},
    )
    sigma2_s = float(np.exp(res.x[0]))
    sigma2_c = float(np.exp(res.x[1]))
    sigma2_e = float(np.exp(res.x[2]))

    # Recompute GLS β and Cov(β) at the MLE, plus BLUPs.
    XtVinvX = np.zeros((p_fixed, p_fixed))
    XtVinvy = np.zeros(p_fixed)
    V_list = []
    for y_s, X_s, inner_ids, inner_unique in blocks_outer:
        n_s = len(y_s)
        V = sigma2_e * np.eye(n_s) + sigma2_s * np.ones((n_s, n_s))
        for c in inner_unique:
            mask = inner_ids == c
            V[np.ix_(mask, mask)] += sigma2_c
        V_list.append(V)
        XtVinvX += X_s.T @ np.linalg.solve(V, X_s)
        XtVinvy += X_s.T @ np.linalg.solve(V, y_s)
    beta_hat = np.linalg.solve(XtVinvX, XtVinvy)
    cov_beta = np.linalg.inv(XtVinvX)

    # BLUPs --------------------------------------------------------------
    # For random-intercept-only 3-level LMM the BLUPs are:
    #   û_school = σ²_s · 1' V_s⁻¹ r_s
    #   û_class  = σ²_c · 1' V_s⁻¹ r_s  (restricted to class indicator)
    school_blups = []
    class_blups = []
    class_blup_keys = []
    class_blup_school = []
    for (y_s, X_s, inner_ids, inner_unique), V in zip(blocks_outer, V_list):
        r = y_s - X_s @ beta_hat
        Vinv_r = np.linalg.solve(V, r)
        u_school = sigma2_s * float(np.sum(Vinv_r))
        school_blups.append(u_school)
        for c in inner_unique:
            mask = inner_ids == c
            u_class = sigma2_c * float(np.sum(Vinv_r[mask]))
            class_blups.append(u_class)
            class_blup_keys.append(c)

    # Build composed BLUP dict: keys are innermost (class) labels mapping
    # to their random intercept (total = school + class).
    # We keep the class-level BLUP (what you'd pass to predict() for
    # prediction within a known class).
    df_inner_to_outer = df.groupby(inner_col, sort=False)[outer_col].first()
    outer_keys = list(df.groupby(outer_col, sort=False).groups.keys())
    school_to_blup = dict(zip(outer_keys, school_blups))

    blup_dict = {}
    blup_rows = []
    for c, u in zip(class_blup_keys, class_blups):
        # Compose the total innermost random intercept for prediction.
        outer_key = df_inner_to_outer[c]
        blup_dict[c] = np.array([u + school_to_blup[outer_key]])
        blup_rows.append({"_cons": u + school_to_blup[outer_key]})
    random_effects_df = pd.DataFrame(blup_rows, index=class_blup_keys)
    random_effects_df.index.name = inner_col

    # Variance components ------------------------------------------------
    vc = {
        f"var(_cons|{outer_col})": sigma2_s,
        f"var(_cons|{inner_col})": sigma2_c,
        "var(Residual)": sigma2_e,
    }

    # Log-likelihood ML
    nll = float(res.fun)
    ll = -nll
    if reml:
        sign, logdet = np.linalg.slogdet(XtVinvX)
        ll_ml = -(nll - 0.5 * logdet - (-0.5 * p_fixed * np.log(2 * np.pi)))
    else:
        ll_ml = ll

    icc_outer = sigma2_s / (sigma2_s + sigma2_c + sigma2_e) if (sigma2_s + sigma2_c + sigma2_e) > 0 else np.nan

    # Package the fit as a MixedResult with _blocks tailored for predict/R².
    # For the three-level case we expose the outermost random-effect
    # variance via ``icc`` (school-level) and keep the full VC dict.
    blocks_proxy = []
    for y_s, X_s, inner_ids, inner_unique in blocks_outer:
        Z_s = np.ones((len(y_s), 1))
        blocks_proxy.append(_GroupBlock(key=None, y=y_s, X=X_s, Z=Z_s, n=len(y_s)))

    return MixedResult(
        fixed_effects=pd.Series(beta_hat, index=fixed_names),
        random_effects=random_effects_df,
        variance_components=vc,
        blups=blup_dict,
        n_obs=n_obs,
        n_groups=len(class_blup_keys),
        icc=icc_outer,
        log_likelihood=ll_ml,
        _se_fixed=pd.Series(np.sqrt(np.diag(cov_beta)), index=fixed_names),
        _cov_fixed=cov_beta,
        _G=np.array([[sigma2_s + sigma2_c]]),
        _sigma2=sigma2_e,
        _blocks=blocks_proxy,
        _group_cols=[inner_col],
        _fixed_names=fixed_names,
        _random_names=random_names,
        _x_fixed=list(x_fixed),
        _x_random=[],
        _y_name=y,
        _method=method,
        _cov_type="three-level-nested",
        _converged=bool(res.success),
        _alpha=alpha,
        _n_cov_params=3,
        _lr_test=None,
        _ranef_se=None,
    )


# ---------------------------------------------------------------------------
# Core negative log-likelihood
# ---------------------------------------------------------------------------


def _profiled_nll(
    theta: np.ndarray,
    blocks: List[_GroupBlock],
    p_fixed: int,
    q_random: int,
    n_total: int,
    reml: bool,
    cov_type: str,
) -> float:
    """Negative (RE)ML log-likelihood profiled over β."""
    n_cov_pars = _n_cov_params(q_random, cov_type)
    G = _unpack_G(theta[:n_cov_pars], q_random, cov_type)
    sigma2 = float(np.exp(theta[n_cov_pars]))

    XtVinvX = np.zeros((p_fixed, p_fixed))
    XtVinvy = np.zeros(p_fixed)
    logdet_sum = 0.0

    # Pass 1: build XtVinvX / XtVinvy & accumulate log|V_j|.
    VinvX_cache = [None] * len(blocks)
    Vinvy_cache = [None] * len(blocks)
    for idx, b in enumerate(blocks):
        V = b.V(G, sigma2)
        try:
            VinvX, logdet = _solve_V(V, b.X)
            Vinvy, _ = _solve_V(V, b.y)
        except np.linalg.LinAlgError:
            return 1e12
        VinvX_cache[idx] = VinvX
        Vinvy_cache[idx] = Vinvy
        XtVinvX += b.X.T @ VinvX
        XtVinvy += b.X.T @ Vinvy
        logdet_sum += logdet

    # GLS β̂(θ)
    try:
        beta = np.linalg.solve(XtVinvX, XtVinvy)
    except np.linalg.LinAlgError:
        return 1e12

    # Pass 2: quadratic form with profiled β
    quad_sum = 0.0
    for idx, b in enumerate(blocks):
        r = b.y - b.X @ beta
        V = b.V(G, sigma2)
        try:
            Vinvr, _ = _solve_V(V, r)
        except np.linalg.LinAlgError:
            return 1e12
        quad_sum += r @ Vinvr

    nll = 0.5 * (logdet_sum + quad_sum + n_total * np.log(2 * np.pi))

    if reml:
        sign, logdet_xtvinvx = np.linalg.slogdet(XtVinvX)
        if sign <= 0:
            return 1e12
        nll += 0.5 * logdet_xtvinvx
        nll -= 0.5 * p_fixed * np.log(2 * np.pi)

    return nll


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mixed(
    data: pd.DataFrame,
    y: str,
    x_fixed: Sequence[str],
    group: "str | Sequence[str]",
    x_random: "Sequence[str] | None" = None,
    cov_type: str = "unstructured",
    method: str = "reml",
    maxiter: int = 400,
    tol: float = 1e-7,
    alpha: float = 0.05,
) -> MixedResult:
    """
    Fit a linear mixed-effects model.

    Parameters
    ----------
    data
        Long-format panel / grouped data.
    y
        Dependent variable column.
    x_fixed
        Fixed-effect regressors (intercept is added automatically).
    group
        Grouping variable.  Pass a list like ``["school", "class"]`` to
        estimate a three-level nested model — the innermost level is
        used as the cluster for the random slopes/intercept; the outer
        levels enter as additional random-intercept blocks.
    x_random
        Random-slope variables.  ``None`` ⇒ random intercept only.
    cov_type
        Parameterisation of the random-effect covariance matrix *G*:
        ``'unstructured'`` (default), ``'diagonal'``, or ``'identity'``.
    method
        ``'reml'`` (default) or ``'ml'``.
    maxiter, tol, alpha
        Optimiser controls and inference significance level.

    Returns
    -------
    MixedResult
    """
    if method not in ("reml", "ml"):
        raise ValueError("method must be 'reml' or 'ml'")
    if cov_type not in ("unstructured", "diagonal", "identity"):
        raise ValueError(
            f"cov_type must be 'unstructured', 'diagonal' or 'identity', got {cov_type!r}"
        )

    group_cols = _as_str_list(group)
    if len(group_cols) > 2:
        raise ValueError(
            "mixed() supports up to two nested grouping levels; deeper hierarchies "
            "must be flattened into a single cluster key first."
        )
    x_fixed = list(x_fixed)
    x_random_cols: List[str] = list(x_random) if x_random is not None else []

    df = _prepare_frame(data, y, x_fixed, group_cols, x_random_cols)

    # Dispatch to the dedicated three-level nested routine when the user
    # passes ``group=[outer, inner]``.  Random slopes are not currently
    # supported through this path — for slopes at the innermost level
    # collapse the nested key yourself and pass a single group.
    if len(group_cols) == 2:
        if x_random_cols:
            raise NotImplementedError(
                "three-level models currently support random intercepts only; "
                "collapse the inner cluster into a unique key and pass "
                "group=<that key> if you need innermost-level random slopes."
            )
        return _fit_three_level_intercept(
            df,
            y=y,
            x_fixed=x_fixed,
            outer_col=group_cols[0],
            inner_col=group_cols[1],
            method=method,
            maxiter=maxiter,
            tol=tol,
            alpha=alpha,
        )

    outer_col = None
    group_fit_col = group_cols[0]

    blocks, fixed_names, random_names = _group_blocks(
        df, y, x_fixed, x_random_cols, group_fit_col
    )

    p_fixed = 1 + len(x_fixed)
    q_random = 1 + len(x_random_cols)
    n_obs = len(df)
    n_groups = len(blocks)
    n_cov_pars = _n_cov_params(q_random, cov_type)

    # Starting values: pooled OLS residuals inform sigma².
    X_all = np.vstack([b.X for b in blocks])
    y_all = np.concatenate([b.y for b in blocks])
    ols_beta, *_ = np.linalg.lstsq(X_all, y_all, rcond=None)
    s2_ols = float(np.var(y_all - X_all @ ols_beta, ddof=max(1, p_fixed)))

    theta_cov0 = _initial_theta(q_random, cov_type, s2_init=0.1 * s2_ols)
    theta0 = np.concatenate([theta_cov0, [np.log(max(0.9 * s2_ols, 1e-6))]])

    reml = method == "reml"

    res = minimize(
        _profiled_nll,
        theta0,
        args=(blocks, p_fixed, q_random, n_obs, reml, cov_type),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": tol, "gtol": tol},
    )
    converged = bool(res.success)

    # Extract solution ------------------------------------------------------
    G_hat = _unpack_G(res.x[:n_cov_pars], q_random, cov_type)
    sigma2_hat = float(np.exp(res.x[n_cov_pars]))

    # GLS β and Cov(β) at the MLE
    XtVinvX = np.zeros((p_fixed, p_fixed))
    XtVinvy = np.zeros(p_fixed)
    Vinv_cache: List[np.ndarray] = []
    for b in blocks:
        V = b.V(G_hat, sigma2_hat)
        VinvX, _ = _solve_V(V, b.X)
        Vinvy, _ = _solve_V(V, b.y)
        XtVinvX += b.X.T @ VinvX
        XtVinvy += b.X.T @ Vinvy
        Vinv_cache.append(np.linalg.solve(V, np.eye(V.shape[0])))
    beta_hat = np.linalg.solve(XtVinvX, XtVinvy)
    cov_beta = np.linalg.inv(XtVinvX)
    se_beta = np.sqrt(np.diag(cov_beta))

    fixed_effects = pd.Series(beta_hat, index=fixed_names)
    se_fixed = pd.Series(se_beta, index=fixed_names)

    # BLUPs and posterior SEs -----------------------------------------------
    blup_rows: List[Dict[str, float]] = []
    blup_dict: Dict[Any, np.ndarray] = {}
    ranef_se_rows: List[Dict[str, float]] = []
    keys: List[Any] = []

    for b, Vinv in zip(blocks, Vinv_cache):
        r = b.y - b.X @ beta_hat
        u_hat = G_hat @ b.Z.T @ Vinv @ r
        # Conditional variance:
        #   Var(u|y) = G - G Z' V^{-1} Z G + G Z' V^{-1} X Cov(β) X' V^{-1} Z G
        ZtVinvZ = b.Z.T @ Vinv @ b.Z
        cond_var = G_hat - G_hat @ ZtVinvZ @ G_hat
        inflate = G_hat @ b.Z.T @ Vinv @ b.X @ cov_beta @ b.X.T @ Vinv @ b.Z @ G_hat
        cond_var = cond_var + inflate
        # Numerical PSD cleanup.
        cond_var = 0.5 * (cond_var + cond_var.T)
        diag = np.clip(np.diag(cond_var), 0.0, None)

        blup_dict[b.key] = u_hat
        blup_rows.append(dict(zip(random_names, u_hat)))
        ranef_se_rows.append(dict(zip(random_names, np.sqrt(diag))))
        keys.append(b.key)

    random_effects_df = pd.DataFrame(blup_rows, index=keys)
    random_effects_df.index.name = group_fit_col
    ranef_se_df = pd.DataFrame(ranef_se_rows, index=keys)
    ranef_se_df.index.name = group_fit_col

    # Variance components ----------------------------------------------------
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
    vc["var(Residual)"] = sigma2_hat

    # (Three-level nesting is handled by a dedicated routine; see
    # ``_fit_three_level_intercept`` earlier in the dispatcher.)

    # Log-likelihood --------------------------------------------------------
    nll_opt = float(res.fun)
    ll = -nll_opt
    if reml:
        # Convert REML objective to ML log-lik for reporting.  REML adds
        # +0.5 log|XtVinvX| and -0.5 p log(2π); remove to get ML.
        sign, logdet = np.linalg.slogdet(XtVinvX)
        ll_ml = -(nll_opt - 0.5 * logdet - (-0.5 * p_fixed * np.log(2 * np.pi)))
    else:
        ll_ml = ll

    # ICC -----------------------------------------------------------------
    sigma2_u0 = float(G_hat[0, 0])
    icc = sigma2_u0 / (sigma2_u0 + sigma2_hat) if (sigma2_u0 + sigma2_hat) > 0 else np.nan

    # LR test vs. pooled OLS (ML likelihood basis) ---------------------------
    resid_ols = y_all - X_all @ ols_beta
    s2_ml_ols = float(np.sum(resid_ols ** 2) / n_obs)
    ll_ols = -0.5 * n_obs * (np.log(2 * np.pi * s2_ml_ols) + 1)
    chi2 = max(2.0 * (ll_ml - ll_ols), 0.0)
    lr_test = {
        "chi2": chi2,
        "df": float(n_cov_pars),
        "p": float(1.0 - stats.chi2.cdf(chi2, n_cov_pars)) if chi2 > 0 else 1.0,
    }

    return MixedResult(
        fixed_effects=fixed_effects,
        random_effects=random_effects_df,
        variance_components=vc,
        blups=blup_dict,
        n_obs=n_obs,
        n_groups=n_groups,
        icc=icc,
        log_likelihood=ll_ml,
        _se_fixed=se_fixed,
        _cov_fixed=cov_beta,
        _G=G_hat,
        _sigma2=sigma2_hat,
        _blocks=blocks,
        _group_cols=[group_fit_col],
        _fixed_names=fixed_names,
        _random_names=random_names,
        _x_fixed=x_fixed,
        _x_random=x_random_cols,
        _y_name=y,
        _method=method,
        _cov_type=cov_type,
        _converged=converged,
        _alpha=alpha,
        _n_cov_params=n_cov_pars + 1,  # + residual variance
        _lr_test=lr_test,
        _ranef_se=ranef_se_df,
    )


__all__ = ["mixed", "MixedResult"]
