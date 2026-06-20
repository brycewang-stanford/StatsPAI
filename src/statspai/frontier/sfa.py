"""
Cross-sectional stochastic frontier estimation — :func:`frontier`.

Supports:

* **Distributions**: half-normal, exponential, truncated-normal.
* **Heteroskedastic u**: ``sigma_u_i = exp(w_i' gamma_u)`` via ``usigma=[...]``.
* **Heteroskedastic v**: ``sigma_v_i = exp(r_i' gamma_v)`` via ``vsigma=[...]``.
* **Inefficiency determinants**: ``mu_i = z_i' delta`` for truncated-normal via
  ``emean=[...]``  (Battese-Coelli 1995 cross-sectional analogue,
  Kumbhakar-Ghosh-McGuckin 1991).
* **Cost / production**: ``cost=True`` flips sign of u in composed error.
* **Technical efficiency**: Battese-Coelli (1988) ``E[exp(-u)|eps]`` or JLMS
  ``exp(-E[u|eps])`` via ``te_method``.
* **Specification tests**: LR test against OLS (absence of inefficiency) using
  mixed chi-bar-squared (Kodde-Palm 1986); LR test of half-normal vs
  truncated-normal; residual skewness diagnostic.

Equivalent to (and more general than) Stata's::

    frontier y x1 x2, distribution(hnormal | exponential | tnormal)
    frontier y x1 x2, cost
    frontier y x1 x2, usigma(w1 w2) vsigma(r1)
    frontier y x1 x2, distribution(tnormal) emean(z1 z2)

and R's ``frontier::sfa()`` / ``sfaR::sfacross()``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ..core.results import EconometricResults
from ..exceptions import (
    ConvergenceFailure,
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)
from . import _core as _fc


def _require_string_option(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise MethodIncompatibility(
            f"`{name}` must be a string option.",
            diagnostics={name: repr(value)},
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


def _coerce_column_list(
    columns: Any,
    name: str,
    *,
    allow_empty: bool = False,
) -> List[str]:
    if isinstance(columns, str):
        out = [columns]
    else:
        try:
            out = list(columns)
        except TypeError as exc:
            raise MethodIncompatibility(
                f"`{name}` must be a column name or a list of column names.",
                diagnostics={name: repr(columns)},
            ) from exc
    if not allow_empty and not out:
        raise MethodIncompatibility(
            f"`{name}` must contain at least one column name.",
            diagnostics={name: out},
        )
    bad = [c for c in out if not isinstance(c, str) or not c]
    if bad:
        raise MethodIncompatibility(
            f"`{name}` must contain only non-empty string column names.",
            diagnostics={name: out, "invalid_columns": bad},
        )
    return out


def _coerce_optional_column_list(columns: Any, name: str) -> Optional[List[str]]:
    if columns is None:
        return None
    return _coerce_column_list(columns, name, allow_empty=True)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class FrontierResult(EconometricResults):
    """Result object returned by :func:`frontier` and :func:`xtfrontier`.

    Extends :class:`~statspai.core.results.EconometricResults` with
    efficiency-score access, LR tests, and bootstrap helpers.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> log_k = rng.normal(0, 1, 150)
    >>> log_l = rng.normal(0, 1, 150)
    >>> u = rng.exponential(0.3, 150)
    >>> v = rng.normal(0, 0.2, 150)
    >>> log_y = 1.0 + 0.4 * log_k + 0.5 * log_l + v - u
    >>> df = pd.DataFrame({"log_y": log_y, "log_k": log_k, "log_l": log_l})
    >>> res = sp.frontier(df, y="log_y", x=["log_k", "log_l"])
    >>> type(res).__name__
    'FrontierResult'
    >>> eff = res.efficiency()
    >>> bool((eff > 0).all() and (eff <= 1.0 + 1e-8).all())
    True
    """

    # ---- Names of diagnostics whose values are per-observation arrays ----
    # The parent class renders every diagnostic literally; we override summary
    # to show an SFA-specific block instead of dumping these arrays.
    _ARRAY_DIAGS = frozenset(
        {
            "sigma_u_i",
            "sigma_v_i",
            "mu_i",
            "eps",
            "efficiency_bc",
            "efficiency_jlms",
            "inefficiency_jlms",
            "efficiency_index",
            "a_it",
            "group_idx",
            "unit_ids",
            "efficiency_bc_unit",
            "efficiency_jlms_unit",
            "efficiency_bc_unit_mean",
            "hessian",
            "vcov",
            "spec",
        }
    )

    def summary(self, alpha: float = 0.05) -> str:
        """Formatted summary table (Stata-style SFA block).

        Overrides :class:`EconometricResults.summary` to hide per-observation
        diagnostic arrays and surface the SFA-specific scalars.
        """
        # Hide array diagnostics from the parent's generic renderer.
        # Use a snapshot + full try/finally so that an exception between
        # the pop and the parent call cannot leave diagnostics half-
        # emptied (previous code could lose diagnostics permanently if
        # an error were raised before the outer try: ran).
        hidden: Dict[str, Any] = {}
        try:
            for k in list(self.diagnostics.keys()):
                if k in self._ARRAY_DIAGS:
                    hidden[k] = self.diagnostics.pop(k)
            base = EconometricResults.summary(self, alpha=alpha)
        finally:
            self.diagnostics.update(hidden)

        # Build the SFA-specific footer.
        mi = self.model_info
        lines = ["", "Variance decomposition:"]
        lines.append("-" * 20)
        if "sigma_u_mean" in mi:  # cross-sectional
            su, sv = mi["sigma_u_mean"], mi["sigma_v_mean"]
        else:
            su, sv = mi.get("sigma_u", float("nan")), mi.get("sigma_v", float("nan"))
        sigma2 = su**2 + sv**2
        lam = su / sv if sv > 0 else float("nan")
        gamma = su**2 / sigma2 if sigma2 > 0 else float("nan")
        lines.append(f"  sigma_u          : {su:.6f}")
        lines.append(f"  sigma_v          : {sv:.6f}")
        lines.append(f"  sigma            : {np.sqrt(sigma2):.6f}")
        lines.append(f"  lambda = su/sv   : {lam:.4f}")
        lines.append(f"  gamma  = su^2/s^2: {gamma:.4f}")
        mean_bc = mi.get("mean_efficiency_bc")
        mean_jlms = mi.get("mean_efficiency_jlms")
        if mean_bc is not None:
            lines.append(f"  mean TE (BC)     : {mean_bc:.4f}")
        if mean_jlms is not None:
            lines.append(f"  mean TE (JLMS)   : {mean_jlms:.4f}")

        lr = self.diagnostics.get("lr_no_inefficiency")
        if lr is not None:
            from . import _core as _fc

            pval = _fc.mixed_chi_bar_pvalue(float(lr), df_boundary=1)
            lines.append("")
            lines.append(
                "LR test vs OLS (H0: sigma_u=0):  "
                f"chi-bar^2(1)={lr:.4f}  p={pval:.4f}"
            )

        return base + "\n" + "\n".join(lines)

    def efficiency(
        self,
        method: Optional[str] = None,
    ) -> pd.Series:
        """Return unit-level technical efficiency scores.

        Parameters
        ----------
        method : {'bc', 'jlms'}, optional
            'bc' (default) : Battese-Coelli (1988) ``E[exp(-u)|eps]``.
            'jlms'         : Jondrow-Lovell-Materov-Schmidt ``exp(-E[u|eps])``.
            If None, uses the default stored at fit time.
        """
        key = self._efficiency_key(method)
        vals = self.diagnostics.get(key)
        if vals is None:
            raise MethodIncompatibility(
                f"Efficiency scores '{key}' not available.",
                recovery_hint=(
                    "Use a fitted frontier result that stores technical "
                    "efficiency diagnostics, or choose a supported method."
                ),
                diagnostics={
                    "method": method,
                    "missing_diagnostic": key,
                    "available": sorted(self.diagnostics),
                },
            )
        # TRE stores a broadcast marginal, not a per-obs posterior; warn
        # once per call so callers don't mistake a constant Series for a
        # degenerate model fit.
        if self.model_info.get("efficiency_kind") == "tre_marginal":
            warnings.warn(
                "TRE efficiency scores are the marginal E[exp(-u)] "
                "broadcast to every observation (posterior E[.|e_i] "
                "integration over alpha_i is not implemented). "
                "res.efficiency().std() will be 0 by construction; "
                "use TFE or BC95 for per-observation scores.",
                UserWarning,
                stacklevel=2,
            )
        idx = self.diagnostics.get("efficiency_index")
        return pd.Series(vals, name=key, index=idx if idx is not None else None)

    def inefficiency(self, method: str = "jlms") -> pd.Series:
        """Return ``E[u|eps]`` (inefficiency), Jondrow et al. (1982)."""
        vals = self.diagnostics.get("inefficiency_jlms")
        if vals is None:
            raise MethodIncompatibility(
                "Inefficiency scores not available.",
                recovery_hint=(
                    "Use a fitted frontier result that stores JLMS "
                    "inefficiency diagnostics."
                ),
                diagnostics={"missing_diagnostic": "inefficiency_jlms"},
            )
        idx = self.diagnostics.get("efficiency_index")
        return pd.Series(vals, name="u_hat", index=idx if idx is not None else None)

    def _efficiency_key(self, method: Optional[str]) -> str:
        if method is None:
            method = self.model_info.get("te_method", "bc")
        _require_string_option(method, "method")
        method = method.lower()
        if method in {"bc", "battese-coelli", "battesecoelli", "bc_mixture"}:
            return "efficiency_bc"
        if method in {"jlms", "jondrow", "jlms_mixture"}:
            return "efficiency_jlms"
        raise MethodIncompatibility(
            f"Unknown TE method: {method!r}",
            recovery_hint="Choose method='bc' or method='jlms'.",
            diagnostics={"method": method, "valid": ["bc", "jlms"]},
        )

    # ------------------------------------------------------------------
    # Out-of-sample prediction
    # ------------------------------------------------------------------

    def predict(  # type: ignore[override]
        self,
        new_data: pd.DataFrame,
        what: str = "frontier",
    ) -> pd.Series:
        """Out-of-sample prediction.

        Parameters
        ----------
        new_data : pandas.DataFrame
            Must contain the frontier regressors and, if the model has
            ``usigma`` / ``vsigma`` / ``emean`` covariates, those columns too.
            For ``what='conditional_*'`` the dependent variable ``y`` must
            also be present so that the composed-error residual can be
            computed and conditioned on.
            Rows with any missing value are dropped.
        what : {'frontier', 'expected_inefficiency', 'expected_efficiency',
                'conditional_inefficiency', 'conditional_efficiency'}
            * ``'frontier'`` — deterministic frontier ``x_new' beta``.
            * ``'expected_inefficiency'`` — marginal ``E[u_new]``.
            * ``'expected_efficiency'`` — marginal ``E[exp(-u_new)]``.
            * ``'conditional_inefficiency'`` — Jondrow posterior
              ``E[u | eps_new]`` where ``eps_new = y_new - x_new'beta``;
              requires ``y`` column in ``new_data``.
            * ``'conditional_efficiency'`` — Battese-Coelli
              ``E[exp(-u) | eps_new]``; requires ``y``.

        Returns
        -------
        pandas.Series
            Indexed by the (post-dropna) rows of ``new_data``.
        """
        if not isinstance(what, str):
            raise MethodIncompatibility(
                "`what` must be a string prediction target.",
                diagnostics={"what": repr(what)},
            )
        what = what.lower()
        valid = {
            "frontier",
            "expected_inefficiency",
            "expected_efficiency",
            "conditional_inefficiency",
            "conditional_efficiency",
        }
        if what not in valid:
            raise MethodIncompatibility(
                f"Unknown what={what!r}.  Valid: {sorted(valid)}.",
                recovery_hint=(
                    "Choose one of: frontier, expected_inefficiency, "
                    "expected_efficiency, conditional_inefficiency, "
                    "conditional_efficiency."
                ),
                diagnostics={"what": what, "valid": sorted(valid)},
            )
        if not isinstance(new_data, pd.DataFrame):
            raise MethodIncompatibility(
                "frontier predict() requires a pandas DataFrame.",
                diagnostics={"data_type": new_data.__class__.__name__},
            )

        regressors = self.data_info.get("regressors", [])
        usigma_cols = self.data_info.get("usigma_cols") or []
        vsigma_cols = self.data_info.get("vsigma_cols") or []
        emean_cols = self.data_info.get("emean_cols") or []

        required = (
            list(regressors) + list(usigma_cols) + list(vsigma_cols) + list(emean_cols)
        )
        needs_y = what.startswith("conditional_")
        if needs_y:
            y_col = self.data_info.get("dep_var")
            if y_col is None or y_col not in new_data.columns:
                raise MethodIncompatibility(
                    f"what={what!r} requires the dependent variable "
                    f"{y_col!r} to be present in new_data.",
                    recovery_hint=(
                        "Include the fitted outcome column when requesting "
                        "conditional frontier predictions."
                    ),
                    diagnostics={"dependent_var": y_col, "what": what},
                )
            required = [y_col] + required
        missing = [c for c in required if c not in new_data.columns]
        if missing:
            raise MethodIncompatibility(
                f"new_data is missing required columns: {missing}",
                recovery_hint=(
                    "Add the fitted frontier, variance, inefficiency, and "
                    "conditional-outcome columns needed for this prediction."
                ),
                diagnostics={"missing_columns": missing, "what": what},
            )
        df_new = new_data[required].dropna().copy()
        idx = df_new.index
        n_new = len(df_new)
        if n_new == 0:
            raise DataInsufficient(
                "All rows dropped after removing missing values.",
                recovery_hint=(
                    "Provide at least one row with complete prediction inputs."
                ),
                diagnostics={"required_columns": required},
            )
        try:
            numeric_required = df_new[required].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "frontier prediction columns must be numeric.",
                diagnostics={"required_columns": required, "error": str(exc)},
            ) from exc
        if not np.isfinite(numeric_required).all():
            raise MethodIncompatibility(
                "frontier prediction columns must be finite.",
                diagnostics={"required_columns": required},
            )

        # --- Rebuild design and pull fitted coefficients ---
        params = self.params
        beta = params.loc[["_cons"] + list(regressors)].to_numpy()
        const = np.ones((n_new, 1))
        X_new = np.concatenate(
            [const, df_new[regressors].to_numpy(dtype=float)],
            axis=1,
        )
        frontier_hat = X_new @ beta

        if what == "frontier":
            return pd.Series(frontier_hat, index=idx, name="frontier")

        # Need sigma_u / sigma_v / mu for new rows.
        # Canonical constant-name comes from _core.const_name_for so the
        # predict() path never diverges from how parameters were named
        # at fit time by build_optional_design().
        u_const_name = _fc.const_name_for("u_") if usigma_cols else "ln_sigma_u"
        sigma_u_new = self._eval_sigma(
            df_new,
            usigma_cols,
            log_sigma_const_name=u_const_name,
            coef_prefix="u_",
        )
        mu_new = self._eval_mu(df_new, emean_cols)
        dist = self.model_info.get("inefficiency_dist", "half-normal")

        if dist == "half-normal":
            E_u = sigma_u_new * np.sqrt(2.0 / np.pi)
        elif dist == "exponential":
            E_u = sigma_u_new
        elif dist == "truncated-normal":
            ratio = mu_new / sigma_u_new
            E_u = mu_new + sigma_u_new * _fc._phi_over_Phi(ratio)
            E_u = np.maximum(E_u, 0.0)
        else:
            raise NumericalInstability(
                f"Unsupported fitted distribution for predict(): {dist!r}",
                recovery_hint=(
                    "Refit with dist='half-normal', 'exponential', or "
                    "'truncated-normal'."
                ),
                diagnostics={"inefficiency_dist": dist},
            )

        if what == "expected_inefficiency":
            return pd.Series(E_u, index=idx, name="expected_inefficiency")

        if what.startswith("conditional_"):
            y_col = self.data_info.get("dep_var")
            y_new = df_new[y_col].to_numpy(dtype=float)
            v_const_name = _fc.const_name_for("v_") if vsigma_cols else "ln_sigma_v"
            sigma_v_new = self._eval_sigma(
                df_new,
                vsigma_cols,
                log_sigma_const_name=v_const_name,
                coef_prefix="v_",
            )
            eps_new = y_new - frontier_hat
            sign = self.model_info.get("sign", -1)
            if dist == "half-normal":
                E_u_cond, TE_bc_cond = _fc.jondrow_halfnormal(
                    eps_new, sigma_v_new, sigma_u_new, sign
                )
            elif dist == "exponential":
                E_u_cond, TE_bc_cond = _fc.jondrow_exponential(
                    eps_new, sigma_v_new, sigma_u_new, sign
                )
            else:
                E_u_cond, TE_bc_cond = _fc.jondrow_truncnormal(
                    eps_new, sigma_v_new, sigma_u_new, mu_new, sign
                )
            if what == "conditional_inefficiency":
                return pd.Series(E_u_cond, index=idx, name="conditional_inefficiency")
            return pd.Series(
                np.clip(TE_bc_cond, 0.0, 1.0),
                index=idx,
                name="conditional_efficiency",
            )

        # Marginal E[exp(-u_new)] for each distribution. Route through the
        # stabilized :func:`_core._battese_coelli_te` for HN / TN (HN is
        # the mu=0 special case of TN) so extreme bootstrap-path sigma
        # values don't overflow the exp term.
        if dist == "half-normal":
            te = _fc._battese_coelli_te(np.zeros_like(sigma_u_new), sigma_u_new)
        elif dist == "exponential":
            # E[exp(-u)] with u ~ Exp(scale=sigma_u): 1/(1 + sigma_u).
            te = 1.0 / (1.0 + sigma_u_new)
        else:  # truncated-normal
            te = _fc._battese_coelli_te(mu_new, sigma_u_new)
        te = np.clip(te, 0.0, 1.0)
        return pd.Series(te, index=idx, name="expected_efficiency")

    def _eval_sigma(
        self,
        df: pd.DataFrame,
        cols: List[str],
        log_sigma_const_name: str,
        coef_prefix: str,
    ) -> np.ndarray:
        """Compute per-row sigma from fitted log-sigma coefficients.

        Handles both the homoskedastic case (single ``ln_sigma_{u,v}`` param)
        and the heteroskedastic case (``{prefix}_cons`` plus ``{prefix}{col}``).
        The constant-name is derived from :func:`_core.const_name_for` so
        this stays consistent with parameter naming at fit time.
        """
        if not cols:
            return np.asarray(
                np.exp(np.full(len(df), float(self.params[log_sigma_const_name]))),
                dtype=float,
            )
        intercept_name = _fc.const_name_for(coef_prefix)
        intercept = float(self.params[intercept_name])
        W = df[cols].to_numpy(dtype=float)
        coefs = np.asarray(
            [self.params[f"{coef_prefix}{c}"] for c in cols],
            dtype=float,
        )
        return np.asarray(np.exp(intercept + W @ coefs), dtype=float)

    def _eval_mu(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        """Compute per-row mu for truncated-normal emean."""
        if "mu" in self.params.index and not cols:
            return np.full(len(df), float(self.params["mu"]))
        if not cols:
            return np.zeros(len(df))  # half-normal / exponential: no mu
        intercept = float(self.params[_fc.const_name_for("mu_")])
        Z = df[cols].to_numpy(dtype=float)
        coefs = np.asarray([self.params[f"mu_{c}"] for c in cols], dtype=float)
        return np.asarray(intercept + Z @ coefs, dtype=float)

    # ------------------------------------------------------------------
    # Marginal effects (BC95 + Caudill-Ford-Gropper)
    # ------------------------------------------------------------------

    def marginal_effects(
        self,
        kind: str = "inefficiency",
        source: str = "emean",
        at: str = "observation",
    ) -> pd.DataFrame:
        """Marginal effects of inefficiency-shifting covariates.

        Parameters
        ----------
        kind : {'inefficiency'}
            Currently only ``E[u]`` marginal effects are supported.
        source : {'emean', 'usigma'}
            ``'emean'``  — derivative wrt the BC95 mean covariates
            ``z`` via ``mu_i = delta'[1, z_i]``. Requires
            ``dist='truncated-normal'`` and a model fitted with ``emean=[...]``.
            ``'usigma'`` — derivative wrt the Caudill-Ford-Gropper (1995)
            ``sigma_u`` covariates ``w`` via ``ln sigma_u_i = gamma'[1, w_i]``.
            Requires a model fitted with ``usigma=[...]``.
        at : {'observation', 'mean', 'ame'}

        Formulas
        --------
        emean (truncated-normal):
            d E[u_i] / d z_ij = delta_j * [1 - (mu/sigma) * phi/Phi - (phi/Phi)^2]
        usigma (half-normal):
            d E[u_i] / d w_ij = gamma_j * sigma_u_i * sqrt(2/pi)
        usigma (exponential):
            d E[u_i] / d w_ij = gamma_j * sigma_u_i
        usigma (truncated-normal):
            d E[u_i] / d w_ij =
                gamma_j * sigma_u_i * [phi/Phi + ratio * phi/Phi
                * (phi/Phi - (-ratio))]
            (chain rule through sigma_u_i = exp(gamma'[1, w_i])).
        """
        kind = _require_string_option(kind, "kind").lower()
        if kind != "inefficiency":
            raise MethodIncompatibility(
                "Only kind='inefficiency' is supported.",
                recovery_hint="Use kind='inefficiency'.",
                diagnostics={"kind": kind, "valid": ["inefficiency"]},
            )

        source = _require_string_option(source, "source").lower()
        at = _require_string_option(at, "at").lower()
        if at not in {"observation", "mean", "ame"}:
            raise MethodIncompatibility(
                "Unknown marginal-effects evaluation point.",
                recovery_hint=("Choose at='observation', at='mean', or at='ame'."),
                diagnostics={
                    "at": at,
                    "valid": ["observation", "mean", "ame"],
                },
            )
        if source == "emean":
            return self._marginal_effects_emean(at=at)
        if source == "usigma":
            return self._marginal_effects_usigma(at=at)
        raise MethodIncompatibility(
            f"Unknown source={source!r}.",
            recovery_hint="Choose source='emean' or source='usigma'.",
            diagnostics={"source": source, "valid": ["emean", "usigma"]},
        )

    def _marginal_effects_emean(self, at: str) -> pd.DataFrame:
        if self.model_info.get("inefficiency_dist") != "truncated-normal":
            raise MethodIncompatibility(
                "marginal_effects(source='emean') requires " "dist='truncated-normal'.",
                recovery_hint=(
                    "Refit with dist='truncated-normal' and emean=[...] "
                    "before requesting emean marginal effects."
                ),
                diagnostics={
                    "source": "emean",
                    "inefficiency_dist": self.model_info.get("inefficiency_dist"),
                },
            )
        emean_cols = self.data_info.get("emean_cols")
        if not emean_cols:
            raise MethodIncompatibility(
                "marginal_effects(source='emean') requires model to have "
                "emean=[...].",
                recovery_hint=(
                    "Refit the frontier with emean=[...] before requesting "
                    "emean marginal effects."
                ),
                diagnostics={"source": "emean", "emean_cols": emean_cols},
            )
        mu_i = np.asarray(self.diagnostics["mu_i"])
        sigma_u_i = np.asarray(self.diagnostics["sigma_u_i"])
        ratio = mu_i / sigma_u_i
        phi_over_Phi = _fc._phi_over_Phi(ratio)
        jac = 1.0 - ratio * phi_over_Phi - phi_over_Phi**2
        deltas = np.array([self.params[f"mu_{c}"] for c in emean_cols])
        effects = jac[:, None] * deltas[None, :]
        effects_df = pd.DataFrame(
            effects,
            columns=emean_cols,
            index=self.diagnostics.get("efficiency_index"),
        )
        return effects_df.mean(axis=0) if at in {"mean", "ame"} else effects_df

    def _marginal_effects_usigma(self, at: str) -> pd.DataFrame:
        usigma_cols = self.data_info.get("usigma_cols")
        if not usigma_cols:
            raise MethodIncompatibility(
                "marginal_effects(source='usigma') requires model to have "
                "usigma=[...].",
                recovery_hint=(
                    "Refit the frontier with usigma=[...] before requesting "
                    "usigma marginal effects."
                ),
                diagnostics={"source": "usigma", "usigma_cols": usigma_cols},
            )
        dist = self.model_info.get("inefficiency_dist", "half-normal")
        sigma_u_i = np.asarray(self.diagnostics["sigma_u_i"])
        # Chain rule: d sigma_u_i / d w_ij = sigma_u_i * gamma_j.
        if dist == "half-normal":
            factor = sigma_u_i * np.sqrt(2.0 / np.pi)
        elif dist == "exponential":
            factor = sigma_u_i  # E[u] = sigma_u for Exp(scale=sigma_u)
        else:  # truncated-normal: E[u] = mu + sigma * phi/Phi(mu/sigma)
            mu_i = np.asarray(self.diagnostics["mu_i"])
            ratio = mu_i / sigma_u_i
            phi_over_Phi = _fc._phi_over_Phi(ratio)
            # d E[u]/d sigma_u = phi/Phi(r) - mu/sigma
            # * (1 - r*phi/Phi - (phi/Phi)^2).
            # Simpler derivation: d/d sigma [mu + sigma * M(mu/sigma)]
            # where M = phi/Phi.
            # = M(r) + sigma * M'(r) * (-mu/sigma^2)
            # = M(r) - (mu/sigma) * M'(r) = M(r) - r * [-r*M - M^2] = M + r^2 M + r*M^2
            # = M * (1 + r^2) + r * M^2
            factor = sigma_u_i * (
                phi_over_Phi * (1.0 + ratio**2) + ratio * phi_over_Phi**2
            )

        gammas = np.array([self.params[f"u_{c}"] for c in usigma_cols])
        effects = factor[:, None] * gammas[None, :]
        effects_df = pd.DataFrame(
            effects,
            columns=usigma_cols,
            index=self.diagnostics.get("efficiency_index"),
        )
        return effects_df.mean(axis=0) if at in {"mean", "ame"} else effects_df

    # ------------------------------------------------------------------
    # Returns to scale (Cobb-Douglas / translog convenience)
    # ------------------------------------------------------------------

    def returns_to_scale(
        self,
        inputs: Optional[List[str]] = None,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Sum of input elasticities (RTS) with Wald test H0: RTS = 1 (CRS).

        Parameters
        ----------
        inputs : list of str, optional
            Input-variable names (should be log-transformed inputs in a
            Cobb-Douglas frontier).  Defaults to ``self.data_info['regressors']``.
        alpha : float, default 0.05

        Returns
        -------
        dict with keys: ``rts``, ``se``, ``statistic``, ``pvalue``,
        ``ci_lower``, ``ci_upper``, ``interpretation``.
        """
        if inputs is None:
            inputs = self.data_info.get("regressors", [])
        alpha = _require_open_unit_float(alpha, "alpha")
        if not inputs:
            raise MethodIncompatibility(
                "No regressors found to compute RTS.",
                recovery_hint=(
                    "Pass explicit input parameter names or fit the frontier "
                    "with frontier regressors."
                ),
                diagnostics={"inputs": inputs},
            )
        vcov = self.diagnostics.get("vcov")
        # Build restriction vector R such that R @ beta = sum of elasticities.
        param_names = self.params.index.tolist()
        R = np.zeros(len(param_names))
        for v in inputs:
            if v not in param_names:
                raise MethodIncompatibility(
                    f"{v!r} is not a fitted parameter.",
                    recovery_hint=(
                        "Pass only coefficient names present in " "result.params."
                    ),
                    diagnostics={
                        "missing_parameter": v,
                        "available_parameters": param_names,
                    },
                )
            R[param_names.index(v)] = 1.0

        rts_hat = float(R @ self.params.to_numpy())
        rts_se = float(np.sqrt(R @ vcov @ R)) if vcov is not None else float("nan")
        stat = (rts_hat - 1.0) / rts_se if rts_se > 0 else float("nan")
        pval = (
            2.0 * (1.0 - stats.norm.cdf(abs(stat)))
            if np.isfinite(stat)
            else float("nan")
        )
        z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
        ci_lo = rts_hat - z_crit * rts_se if np.isfinite(rts_se) else float("nan")
        ci_hi = rts_hat + z_crit * rts_se if np.isfinite(rts_se) else float("nan")
        if rts_hat > 1.0 + z_crit * rts_se:
            verdict = "increasing returns to scale (IRS)"
        elif rts_hat < 1.0 - z_crit * rts_se:
            verdict = "decreasing returns to scale (DRS)"
        else:
            verdict = "fail to reject constant returns (CRS)"
        return {
            "rts": rts_hat,
            "se": rts_se,
            "statistic": stat,
            "pvalue": pval,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "interpretation": verdict,
        }

    def lr_test_no_inefficiency(self) -> Dict[str, float]:
        """One-sided LR test ``H0: sigma_u = 0`` (mixed chi-bar squared)."""
        stat = self.diagnostics.get("lr_no_inefficiency")
        if stat is None:
            return {"statistic": np.nan, "pvalue": np.nan, "df": np.nan}
        pval = _fc.mixed_chi_bar_pvalue(stat, df_boundary=1)
        return {"statistic": float(stat), "pvalue": float(pval), "df": 1}

    def efficiency_ci(
        self,
        alpha: float = 0.05,
        B: int = 500,
        method: Optional[str] = None,
        seed: Optional[int] = 0,
    ) -> pd.DataFrame:
        """Parametric-bootstrap CI for unit-level efficiency scores.

        Draws ``(u_b, v_b) ~`` posterior predictive using the fitted
        variance parameters, then recomputes the Jondrow posterior for
        the resampled composed error.  Returns a DataFrame indexed like
        :meth:`efficiency` with columns ``['point', 'lower', 'upper']``.
        """
        alpha = _require_open_unit_float(alpha, "alpha")
        B = _require_int_at_least(B, "B", 2)
        point = self.efficiency(method=method).to_numpy()
        sigma_u_i = np.asarray(self.diagnostics.get("sigma_u_i"))
        sigma_v_i = np.asarray(self.diagnostics.get("sigma_v_i"))
        mu_i = self.diagnostics.get("mu_i")
        dist = self.model_info.get("inefficiency_dist", "half-normal")
        sign = self.model_info.get("sign", -1)
        eps = np.asarray(self.diagnostics.get("eps"))
        if eps.size == 0:
            raise DataInsufficient(
                "eps not stored; cannot bootstrap.",
                recovery_hint=(
                    "Use a fitted frontier result that retains residual "
                    "diagnostics before requesting efficiency_ci()."
                ),
                diagnostics={"missing_diagnostic": "eps"},
            )
        rng = np.random.default_rng(seed)
        n = eps.size
        sims = np.empty((B, n))
        for b in range(B):
            # Redraw posterior predictive u, v and reconstruct eps_b.
            if dist == "half-normal":
                u_sim = np.abs(rng.normal(0.0, sigma_u_i))
            elif dist == "exponential":
                u_sim = rng.exponential(sigma_u_i)
            else:  # truncated-normal
                u_sim = _draw_truncated_normal(mu_i, sigma_u_i, rng)
            v_sim = rng.normal(0.0, sigma_v_i)
            eps_b = v_sim + sign * u_sim
            # Posterior E[u|eps_b] under fitted distribution.
            if dist == "half-normal":
                _, te_bc = _fc.jondrow_halfnormal(eps_b, sigma_v_i, sigma_u_i, sign)
            elif dist == "exponential":
                _, te_bc = _fc.jondrow_exponential(eps_b, sigma_v_i, sigma_u_i, sign)
            else:
                _, te_bc = _fc.jondrow_truncnormal(
                    eps_b, sigma_v_i, sigma_u_i, np.asarray(mu_i), sign
                )
            sims[b] = te_bc
        lower = np.quantile(sims, alpha / 2.0, axis=0)
        upper = np.quantile(sims, 1.0 - alpha / 2.0, axis=0)
        idx = self.diagnostics.get("efficiency_index")
        return pd.DataFrame(
            {"point": point, "lower": lower, "upper": upper},
            index=idx if idx is not None else None,
        )


def _refit_bootstrap(
    df_b: pd.DataFrame,
    y: str,
    x: List[str],
    dist: str,
    cost: bool,
    usigma: Optional[List[str]],
    vsigma: Optional[List[str]],
    emean: Optional[List[str]],
    spec: "_FrontierSpec",
    start: np.ndarray,
    maxiter: int,
    tol: float,
) -> np.ndarray:
    """Re-fit the frontier model on one bootstrap sample; return theta-hat.

    Returns an all-NaN vector if the bootstrap replica fails to converge.
    Caller must filter NaN rows before computing the sample covariance,
    otherwise the variance estimate is downward-biased toward the point
    estimate (old behavior returned ``start`` = theta_hat, which silently
    collapsed bootstrap SEs on pathological draws).
    """
    try:
        res = frontier(
            df_b,
            y=y,
            x=x,
            dist=dist,
            cost=cost,
            usigma=usigma,
            vsigma=vsigma,
            emean=emean,
            vce="oim",  # fast SE; we only need the point estimate here
            maxiter=maxiter,
            tol=tol,
            start=start,
        )
        return np.asarray(res.params.to_numpy(), dtype=float)
    except Exception:
        return np.full(spec.k_total, np.nan)


def _draw_truncated_normal(
    mu: Any,
    sigma: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw u ~ N^+(mu, sigma^2) truncated at 0 (inverse-CDF)."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    # Handle broadcasting.
    shape = np.broadcast(mu, sigma).shape
    u = rng.uniform(size=shape)
    lo = stats.norm.cdf(-mu / sigma)
    p = lo + u * (1.0 - lo)
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return np.asarray(mu + sigma * stats.norm.ppf(p), dtype=float)


# ---------------------------------------------------------------------------
# Packing / unpacking of parameter vectors
# ---------------------------------------------------------------------------


@dataclass
class _FrontierSpec:
    """Compact description of the parameter vector layout."""

    k_beta: int
    k_gamma_u: int
    k_gamma_v: int
    k_delta_mu: int  # 0 if no mu (half-normal / exponential)
    has_emean: bool  # True if mu varies with covariates
    has_usigma: bool
    has_vsigma: bool
    dist: str

    @property
    def k_total(self) -> int:
        return self.k_beta + self.k_gamma_u + self.k_gamma_v + self.k_delta_mu

    def slices(self) -> Tuple[slice, slice, slice, slice]:
        a = slice(0, self.k_beta)
        b = slice(self.k_beta, self.k_beta + self.k_gamma_u)
        c = slice(b.stop, b.stop + self.k_gamma_v)
        d = slice(c.stop, c.stop + self.k_delta_mu)
        return a, b, c, d


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def frontier(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    *,
    dist: str = "half-normal",
    cost: bool = False,
    usigma: Optional[List[str]] = None,
    vsigma: Optional[List[str]] = None,
    emean: Optional[List[str]] = None,
    te_method: str = "bc",
    vce: str = "oim",
    cluster: Optional[str] = None,
    B: int = 400,
    seed: Optional[int] = None,
    maxiter: int = 2000,
    tol: float = 1e-10,
    alpha: float = 0.05,
    start: Optional[np.ndarray] = None,
) -> FrontierResult:
    """Estimate a cross-sectional stochastic frontier model by ML.

    Parameters
    ----------
    data : pandas.DataFrame
        Cross-sectional data.  Rows with missing values in any referenced
        column are dropped.
    y : str
        Dependent variable (output for production, cost for cost frontier).
    x : list of str
        Frontier regressors (a constant is added automatically).
    dist : {'half-normal', 'exponential', 'truncated-normal'}
        Distribution of the inefficiency term ``u``.
    cost : bool, default False
        If True, estimate a cost frontier (composed error ``v + u``).
    usigma : list of str, optional
        Columns parameterizing ``ln sigma_u_i = gamma_u' [1, w_i]``
        (Caudill-Ford-Gropper 1995).
    vsigma : list of str, optional
        Columns parameterizing ``ln sigma_v_i = gamma_v' [1, r_i]`` (Wang 2002).
    emean : list of str, optional
        Columns parameterizing ``mu_i = delta' [1, z_i]`` for the truncated
        normal (Battese-Coelli 1995; Kumbhakar-Ghosh-McGuckin 1991).  Requires
        ``dist='truncated-normal'``.
    te_method : {'bc', 'jlms'}, default 'bc'
        Default technical-efficiency formula accessed via ``.efficiency()``.
    vce : {'oim', 'opg', 'robust'}, default 'oim'
        Variance-covariance estimator:
        ``'oim'``    — observed information matrix (inverse numerical Hessian).
        ``'opg'``    — outer product of gradients (Berndt-Hall-Hall-Hausman).
        ``'robust'`` — sandwich ``H^{-1} (S' S) H^{-1}`` (White 1982).
    cluster : str, optional
        Cluster variable for cluster-robust SE (Liang-Zeger 1986).  When
        specified, implies ``vce='robust'`` aggregated over clusters.
    maxiter : int, default 2000
    tol : float, default 1e-10
    alpha : float, default 0.05
    start : ndarray, optional
        User-supplied starting values for the full parameter vector.

    Returns
    -------
    :class:`FrontierResult`

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> log_k = rng.normal(0, 1, n)
    >>> log_l = rng.normal(0, 1, n)
    >>> v = rng.normal(0, 0.2, n)
    >>> u = np.abs(rng.normal(0, 0.3, n))   # one-sided inefficiency
    >>> df = pd.DataFrame({
    ...     'log_y': 0.3 * log_k + 0.6 * log_l + v - u,
    ...     'log_k': log_k,
    ...     'log_l': log_l,
    ... })
    >>> res = sp.frontier(df, y='log_y', x=['log_k', 'log_l'])
    >>> te = res.efficiency()               # firm-level technical efficiency
    >>> bool(0.0 < te.mean() <= 1.0)
    True
    >>> lr = res.lr_test_no_inefficiency()  # H0: no inefficiency (sigma_u = 0)
    >>> sorted(lr)
    ['df', 'pvalue', 'statistic']
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            "frontier() requires a pandas DataFrame.",
            diagnostics={"data_type": data.__class__.__name__},
        )
    y = _require_string_option(y, "y")
    x = _coerce_column_list(x, "x")
    usigma = _coerce_optional_column_list(usigma, "usigma")
    vsigma = _coerce_optional_column_list(vsigma, "vsigma")
    emean = _coerce_optional_column_list(emean, "emean")
    if cluster is not None:
        cluster = _require_string_option(cluster, "cluster")
    te_method = _require_string_option(te_method, "te_method").lower()
    if te_method not in {"bc", "jlms"}:
        raise MethodIncompatibility(
            f"Unknown te_method={te_method!r}.",
            recovery_hint="Choose te_method='bc' or te_method='jlms'.",
            diagnostics={"te_method": te_method, "valid": ["bc", "jlms"]},
        )
    B = _require_int_at_least(B, "B", 2)
    maxiter = _require_int_at_least(maxiter, "maxiter", 1)
    tol = _require_positive_float(tol, "tol")
    _require_open_unit_float(alpha, "alpha")

    dist = _require_string_option(dist, "dist").lower().replace("_", "-")
    if dist not in {"half-normal", "exponential", "truncated-normal"}:
        raise MethodIncompatibility(
            f"Unknown distribution: {dist!r}.",
            recovery_hint=(
                "Choose dist='half-normal', dist='exponential', or "
                "dist='truncated-normal'."
            ),
            diagnostics={
                "dist": dist,
                "valid": ["half-normal", "exponential", "truncated-normal"],
            },
        )
    if emean is not None and dist != "truncated-normal":
        raise MethodIncompatibility(
            "emean=... requires dist='truncated-normal'.",
            recovery_hint=("Use dist='truncated-normal' or remove emean=[...]."),
            diagnostics={"dist": dist, "emean": emean},
        )

    vce = _require_string_option(vce, "vce").lower()
    if vce not in {"oim", "opg", "robust", "bootstrap"}:
        raise MethodIncompatibility(
            f"Unknown vce={vce!r}.",
            recovery_hint=(
                "Choose vce='oim', vce='opg', vce='robust', or " "vce='bootstrap'."
            ),
            diagnostics={
                "vce": vce,
                "valid": ["oim", "opg", "robust", "bootstrap"],
            },
        )
    if cluster is not None and vce == "oim":
        vce = "robust"

    required = [y] + list(x)
    for opt in (usigma, vsigma, emean):
        if opt:
            required += list(opt)
    if cluster is not None and cluster not in required:
        required.append(cluster)
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"frontier() data is missing required columns: {missing}",
            recovery_hint=(
                "Add the outcome, frontier regressors, variance covariates, "
                "inefficiency covariates, or cluster column referenced by "
                "the call."
            ),
            diagnostics={
                "missing_columns": missing,
                "required_columns": required,
            },
        )
    df = data[required].dropna().copy()
    n = len(df)
    if n < len(x) + 3:
        raise DataInsufficient(
            "Too few observations for frontier estimation.",
            recovery_hint=(
                "Provide more complete rows or simplify the frontier "
                "regressor specification."
            ),
            diagnostics={
                "n_complete": n,
                "minimum": len(x) + 3,
                "required_columns": required,
            },
        )

    sign = 1 if cost else -1

    y_vec, X_mat, beta_names = _fc.build_design(df, y, x, add_constant=True)
    W_mat, w_names = _fc.build_optional_design(df, usigma, True, prefix="u_")
    R_mat, r_names = _fc.build_optional_design(df, vsigma, True, prefix="v_")
    Z_mat, z_names = _fc.build_optional_design(df, emean, True, prefix="mu_")

    # Parameter layout
    k_beta = X_mat.shape[1]
    k_gamma_u = W_mat.shape[1] if W_mat is not None else 1
    k_gamma_v = R_mat.shape[1] if R_mat is not None else 1
    if dist == "truncated-normal":
        k_delta_mu = Z_mat.shape[1] if Z_mat is not None else 1
    else:
        k_delta_mu = 0

    spec = _FrontierSpec(
        k_beta=k_beta,
        k_gamma_u=k_gamma_u,
        k_gamma_v=k_gamma_v,
        k_delta_mu=k_delta_mu,
        has_emean=emean is not None,
        has_usigma=usigma is not None,
        has_vsigma=vsigma is not None,
        dist=dist,
    )
    sl_beta, sl_gu, sl_gv, sl_dm = spec.slices()

    # ---------------------- Log-likelihood ----------------------

    def _unpack(
        theta: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        beta = theta[sl_beta]
        gamma_u = theta[sl_gu]
        gamma_v = theta[sl_gv]
        delta = theta[sl_dm] if k_delta_mu > 0 else None

        sigma_u = _fc.evaluate_sigma(
            gamma_u, W_mat, gamma_u[0] if W_mat is None else 0.0, n
        )
        sigma_v = _fc.evaluate_sigma(
            gamma_v, R_mat, gamma_v[0] if R_mat is None else 0.0, n
        )
        if delta is not None:
            if Z_mat is None:
                mu = np.full(n, delta[0])
            else:
                mu = Z_mat @ delta
        else:
            mu = None
        return (
            np.asarray(beta, dtype=float),
            np.asarray(sigma_u, dtype=float),
            np.asarray(sigma_v, dtype=float),
            None if mu is None else np.asarray(mu, dtype=float),
        )

    def per_obs_loglik(theta: np.ndarray) -> np.ndarray:
        """Return vector of per-observation log-likelihoods (length n)."""
        beta, sigma_u, sigma_v, mu = _unpack(theta)
        eps = y_vec - X_mat @ beta
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if dist == "half-normal":
                ll = _fc.loglik_halfnormal(eps, sigma_v, sigma_u, sign)
            elif dist == "exponential":
                ll = _fc.loglik_exponential(eps, sigma_v, sigma_u, sign)
            else:
                assert mu is not None
                ll = _fc.loglik_truncated_normal(eps, sigma_v, sigma_u, mu, sign)
        return np.asarray(ll, dtype=float)

    def neg_loglik(theta: np.ndarray) -> float:
        if not np.all(np.isfinite(theta)):
            return 1e20
        beta, sigma_u, sigma_v, mu = _unpack(theta)
        # Guard against pathological sigma (optimizer excursions).
        if (
            np.any(sigma_u <= 1e-8)
            or np.any(sigma_v <= 1e-8)
            or np.any(sigma_u > 1e6)
            or np.any(sigma_v > 1e6)
        ):
            return 1e20
        ll = per_obs_loglik(theta)
        if not np.isfinite(ll).all():
            return 1e20
        return -float(ll.sum())

    # ---------------------- Starting values ----------------------

    if start is None:
        beta0, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
        resid0 = y_vec - X_mat @ beta0
        sigma0 = float(np.std(resid0))
        sigma0 = max(sigma0, 1e-3)

        ln_sv0 = np.log(sigma0 * 0.7)
        ln_su0 = np.log(sigma0 * 0.7)

        theta0_parts = [beta0]
        if W_mat is None:
            theta0_parts.append(np.array([ln_su0]))
        else:
            tmp = np.zeros(k_gamma_u)
            tmp[0] = ln_su0
            theta0_parts.append(tmp)
        if R_mat is None:
            theta0_parts.append(np.array([ln_sv0]))
        else:
            tmp = np.zeros(k_gamma_v)
            tmp[0] = ln_sv0
            theta0_parts.append(tmp)
        if k_delta_mu > 0:
            if Z_mat is None:
                theta0_parts.append(np.array([0.0]))
            else:
                theta0_parts.append(np.zeros(k_delta_mu))
        start = np.concatenate(theta0_parts)
    else:
        try:
            start = np.asarray(start, dtype=float).copy()
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "start must be numeric.",
                diagnostics={"start_type": start.__class__.__name__},
            ) from exc
        if start.size != spec.k_total:
            raise MethodIncompatibility(
                f"start has wrong length: got {start.size}, "
                f"expected {spec.k_total}.",
                recovery_hint=(
                    "Pass one starting value for each frontier, variance, "
                    "and distribution parameter in the fitted specification."
                ),
                diagnostics={
                    "actual_length": int(start.size),
                    "expected_length": int(spec.k_total),
                },
            )
        if not np.isfinite(start).all():
            raise MethodIncompatibility(
                "start must contain only finite values.",
                diagnostics={"start": start.tolist()},
            )

    # ---------------------- Optimize ----------------------

    # Bounds: loose on betas/mu, tight on log-sigma parameters to keep
    # sigma in a numerically sensible range ~ [e^-12, e^5] ~ [6e-6, 150].
    bounds = []
    for _ in range(k_beta):
        bounds.append((-1e6, 1e6))
    # ln sigma_u block
    bounds.append((-12.0, 5.0))
    for _ in range(k_gamma_u - 1):
        bounds.append((-8.0, 8.0))
    # ln sigma_v block
    bounds.append((-12.0, 5.0))
    for _ in range(k_gamma_v - 1):
        bounds.append((-8.0, 8.0))
    # mu block (truncated-normal only)
    if k_delta_mu > 0:
        bounds.append((-50.0, 50.0))
        for _ in range(k_delta_mu - 1):
            bounds.append((-50.0, 50.0))

    def _fit_from(start_vec: np.ndarray) -> Any:
        return minimize(
            neg_loglik,
            start_vec,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": tol, "gtol": tol},
        )

    result = _fit_from(start)

    # Multi-start safety net for truncated-normal (Waldman 1982):
    # TN is notoriously prone to flat LL near mu=0; try a second start
    # with small positive mu to escape the spurious plateau.
    if dist == "truncated-normal":
        alt_start = start.copy()
        alt_start[sl_dm.start] = 0.5  # perturb intercept of mu block
        alt_result = _fit_from(alt_start)
        if -alt_result.fun > -result.fun + 1e-4:
            result = alt_result

    theta_hat = result.x
    ll_val = -neg_loglik(theta_hat)
    beta_hat, sigma_u_i, sigma_v_i, mu_i = _unpack(theta_hat)

    # ---------------------- Standard errors ----------------------

    H = _fc.numerical_hessian(neg_loglik, theta_hat)
    vcov_oim = _fc.safe_invert_hessian(H)
    if vce == "oim":
        vcov = vcov_oim
    elif vce == "bootstrap":
        # Resample rows (or clusters) with replacement, refit B times.
        rng_boot = np.random.default_rng(seed)
        estimates = np.empty((B, spec.k_total))
        n_df = len(df)
        if cluster is None:
            # Row-level bootstrap
            for b in range(B):
                idx_b = rng_boot.integers(0, n_df, n_df)
                df_b = df.iloc[idx_b].reset_index(drop=True)
                res_b = _refit_bootstrap(
                    df_b,
                    y,
                    x,
                    dist,
                    cost,
                    usigma,
                    vsigma,
                    emean,
                    spec,
                    start=theta_hat,
                    maxiter=maxiter,
                    tol=tol,
                )
                estimates[b] = res_b
        else:
            # Cluster-level bootstrap (preserves within-cluster dep.)
            clusters = df[cluster].unique()
            for b in range(B):
                sampled = rng_boot.choice(clusters, size=len(clusters), replace=True)
                df_b = pd.concat(
                    [df[df[cluster] == c] for c in sampled], ignore_index=True
                )
                res_b = _refit_bootstrap(
                    df_b,
                    y,
                    x,
                    dist,
                    cost,
                    usigma,
                    vsigma,
                    emean,
                    spec,
                    start=theta_hat,
                    maxiter=maxiter,
                    tol=tol,
                )
                estimates[b] = res_b
        # Filter failed replicates (all-NaN rows) before computing variance.
        # Using `start`=theta_hat as a fallback would pull Var() toward zero.
        valid_mask = np.isfinite(estimates).all(axis=1)
        n_valid = int(valid_mask.sum())
        if n_valid < max(5, B // 10):
            raise ConvergenceFailure(
                f"Bootstrap converged on only {n_valid}/{B} replicates; "
                "variance estimate is unreliable. Try vce='robust' or "
                "larger B, or check for near-boundary sigma.",
                recovery_hint=(
                    "Try vce='robust', increase B, or respecify a model "
                    "away from near-boundary variance parameters."
                ),
                diagnostics={"valid_replicates": n_valid, "B": B},
            )
        if n_valid < B:
            warnings.warn(
                f"Bootstrap: {B - n_valid}/{B} replicates failed; "
                f"SE computed from {n_valid} valid draws.",
                RuntimeWarning,
                stacklevel=2,
            )
        vcov = np.cov(estimates[valid_mask], rowvar=False)
    else:
        scores = _fc.per_obs_scores(per_obs_loglik, theta_hat)
        if vce == "opg":
            OPG = scores.T @ scores
            vcov = _fc.safe_invert_hessian(OPG)
        else:  # 'robust' or cluster
            cluster_idx = None
            if cluster is not None:
                cluster_idx = pd.Categorical(df[cluster]).codes.astype(int)
            vcov = _fc.robust_vcov(H, scores, cluster_idx=cluster_idx)
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))

    # ---------------------- Efficiency scores ----------------------

    eps_hat = y_vec - X_mat @ beta_hat
    if dist == "half-normal":
        E_u, TE_bc = _fc.jondrow_halfnormal(eps_hat, sigma_v_i, sigma_u_i, sign)
    elif dist == "exponential":
        E_u, TE_bc = _fc.jondrow_exponential(eps_hat, sigma_v_i, sigma_u_i, sign)
    else:
        assert mu_i is not None
        E_u, TE_bc = _fc.jondrow_truncnormal(eps_hat, sigma_v_i, sigma_u_i, mu_i, sign)
    TE_jlms = np.clip(np.exp(-E_u), 0.0, 1.0)

    # ---------------------- Specification tests ----------------------

    # OLS log-likelihood (H0: no inefficiency).
    resid_ols = y_vec - X_mat @ np.linalg.lstsq(X_mat, y_vec, rcond=None)[0]
    sigma_ols = np.std(resid_ols, ddof=0)
    ll_ols = np.sum(stats.norm.logpdf(resid_ols, loc=0.0, scale=max(sigma_ols, 1e-12)))
    lr_stat_noineff = _fc.lr_test_statistic(ll_val, ll_ols)

    # ---------------------- Assemble result ----------------------

    param_names = list(beta_names)
    if W_mat is None:
        param_names.append("ln_sigma_u")
    else:
        param_names.extend(w_names)
    if R_mat is None:
        param_names.append("ln_sigma_v")
    else:
        param_names.extend(r_names)
    if k_delta_mu > 0:
        if Z_mat is None:
            param_names.append("mu")
        else:
            param_names.extend(z_names)

    params = pd.Series(theta_hat, index=param_names)
    std_errors = pd.Series(se, index=param_names)

    # Summary scalars (for display).
    sigma_u_mean = float(np.mean(sigma_u_i))
    sigma_v_mean = float(np.mean(sigma_v_i))
    sigma_total = float(np.sqrt(sigma_u_mean**2 + sigma_v_mean**2))
    lam_mean = sigma_u_mean / sigma_v_mean if sigma_v_mean > 0 else np.nan
    gamma = sigma_u_mean**2 / (sigma_u_mean**2 + sigma_v_mean**2)

    return FrontierResult(
        params=params,
        std_errors=std_errors,
        model_info={
            "model_type": f"Stochastic Frontier ({'Cost' if cost else 'Production'})",
            "method": f"ML, {dist}",
            "inefficiency_dist": dist,
            "cost": cost,
            "sign": sign,
            "te_method": te_method,
            "vce": vce if cluster is None else f"cluster({cluster})",
            "has_usigma": usigma is not None,
            "has_vsigma": vsigma is not None,
            "has_emean": emean is not None,
            "sigma_u_mean": sigma_u_mean,
            "sigma_v_mean": sigma_v_mean,
            "sigma": sigma_total,
            "lambda": lam_mean,
            "gamma": gamma,
            "mean_efficiency_bc": float(np.mean(TE_bc)),
            "mean_efficiency_jlms": float(np.mean(TE_jlms)),
            "converged": bool(result.success),
        },
        data_info={
            "n_obs": n,
            "dep_var": y,
            "regressors": list(x),
            "usigma_cols": list(usigma) if usigma else None,
            "vsigma_cols": list(vsigma) if vsigma else None,
            "emean_cols": list(emean) if emean else None,
            "df_resid": max(n - spec.k_total, 1),
        },
        diagnostics={
            "log_likelihood": float(ll_val),
            "ll_ols": float(ll_ols),
            "lr_no_inefficiency": float(lr_stat_noineff),
            "aic": float(-2.0 * ll_val + 2.0 * spec.k_total),
            "bic": float(-2.0 * ll_val + np.log(n) * spec.k_total),
            "sigma_u_i": sigma_u_i,
            "sigma_v_i": sigma_v_i,
            "mu_i": mu_i,
            "eps": eps_hat,
            "efficiency_bc": TE_bc,
            "efficiency_jlms": TE_jlms,
            "inefficiency_jlms": E_u,
            "efficiency_index": df.index.to_numpy(),
            "residual_skewness": _fc.ols_residual_skewness(resid_ols),
            "hessian": H,
            "vcov": vcov,
            "spec": spec,
        },
    )


__all__ = ["frontier", "FrontierResult"]
