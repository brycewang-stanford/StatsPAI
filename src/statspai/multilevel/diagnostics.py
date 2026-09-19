"""
Diagnostics for mixed models: intra-class correlation, information
criteria, and the Nakagawa-Schielzeth R² (delegated to the
``MixedResult`` / ``MEGLMResult`` classes themselves).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy import special, stats

from .._result_serialize import ResultProtocolMixin


@dataclass
class ICCResult(ResultProtocolMixin):
    """Container for an ICC estimate with a delta-method Wald CI."""

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    alpha: float

    def summary(self) -> str:
        return (
            f"ICC = {self.estimate:.4f}  (SE = {self.se:.4f}, "
            f"{100 * (1 - self.alpha):.0f}% CI [{self.ci_lower:.4f}, "
            f"{self.ci_upper:.4f}])"
        )

    # Allow ``float(icc(...))`` to recover the point estimate.
    def __float__(self) -> float:
        return float(self.estimate)


_LATENT_LOGIT_VAR = float(np.pi**2 / 3.0)


def _log_var_factor(cov_type: str) -> float:
    """d log var(_cons) / d theta_cov for a single random intercept.

    ``unstructured`` packs the log of the Cholesky diagonal (log sd),
    ``identity`` / ``diagonal`` the log variance.
    """
    return 2.0 if cov_type == "unstructured" else 1.0


def _pack_intercept_theta(var_u: float, cov_type: str) -> float:
    return float(np.log(var_u) / _log_var_factor(cov_type))


def _mixed_vc_cov(result: Any) -> Optional[np.ndarray]:
    """Observed-information covariance of (theta_cov, log sigma2_e).

    Inverse numerical Hessian of the (RE)ML criterion profiled over the
    fixed effects -- for ML this is the variance-parameter block of the
    full inverse information, for REML the inverse REML Hessian, i.e. what
    Stata ``mixed`` reports in ``e(V)`` for its log-sd parameters.
    """
    from .glmm import _numerical_oim_cov
    from .lmm import _profiled_nll

    blocks = getattr(result, "_blocks", None)
    if not blocks:
        return None
    cov_type = result._cov_type
    theta = np.array(
        [
            _pack_intercept_theta(float(result._G[0, 0]), cov_type),
            np.log(float(result._sigma2)),
        ]
    )
    reml = str(result._method).lower() == "reml"
    return _numerical_oim_cov(
        lambda th: _profiled_nll(
            th,
            blocks,
            len(result.fixed_effects),
            1,
            int(result.n_obs),
            reml,
            cov_type,
        ),
        theta,
    )


def _logit_ci(rho: float, se: float, alpha: float) -> tuple:
    """Wald interval on the logit scale, mapped back (Stata ``estat icc``)."""
    if not (0.0 < rho < 1.0) or not np.isfinite(se):
        return np.nan, np.nan
    z = stats.norm.ppf(1 - alpha / 2)
    lg = np.log(rho / (1.0 - rho))
    half = z * se / (rho * (1.0 - rho))
    return float(special.expit(lg - half)), float(special.expit(lg + half))


def icc(
    result: Any,
    component: str = "_cons",
    alpha: float = 0.05,
    n_boot: int = 0,
    seed: Optional[int] = None,
) -> ICCResult:
    """
    Intra-class correlation for a fitted random-intercept model.

    Reproduces Stata ``estat icc``:

    * after :func:`statspai.mixed` (and ``meglm(family='gaussian')``)
      ``ρ = σ²_u / (σ²_u + σ²_e)``;
    * after :func:`statspai.melogit` / :func:`statspai.meologit` the latent
      -variable ICC ``ρ = σ²_u / (σ²_u + π²/3)``;
    * after a three-level nested :func:`statspai.mixed` fit, the level ICCs
      ``σ²_s / total`` (``component`` = outer level) and
      ``(σ²_s + σ²_c) / total`` (inner level).

    The standard error is the delta method on the observed-information
    covariance of the variance parameters (the inverse Hessian of the
    fitted (RE)ML / Laplace criterion), and the confidence interval is a
    Wald interval on the logit scale mapped back to (0, 1) -- both as
    ``estat icc`` reports them.

    Parameters
    ----------
    result
        A ``MixedResult`` from :func:`statspai.mixed` or an
        ``MEGLMResult`` from ``melogit`` / ``meologit`` /
        ``meglm(family='gaussian')``, with a random intercept only.
    component
        Random-intercept name (``"_cons"``); for a three-level fit the
        grouping column of the level whose ICC is wanted.
    alpha
        Significance level for the confidence interval.  Default 0.05.
    n_boot
        Parametric-bootstrap intervals are not implemented; must be 0.
    seed
        Unused (reserved for the bootstrap).

    Returns
    -------
    ICCResult

    Raises
    ------
    MethodIncompatibility
        For random-slope models (the ICC depends on the covariates; Stata
        refuses too) and for count / gamma GLMMs, which have no latent
        residual variance on the linear-predictor scale.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> g = np.repeat(np.arange(30), 10)
    >>> x = rng.normal(size=300)
    >>> y = 2.0 + 0.5 * x + rng.normal(0, 1.0, 30)[g] + rng.normal(0, 0.5, 300)
    >>> df = pd.DataFrame({"y": y, "x": x, "school": g})
    >>> res = sp.mixed(df, y="y", x_fixed=["x"], group="school")
    >>> rho = sp.icc(res)
    >>> bool(0.0 <= float(rho) <= 1.0)  # share of variance at the school level
    True
    """
    from ..exceptions import MethodIncompatibility

    if not hasattr(result, "variance_components"):
        raise TypeError("icc() expects a MixedResult-like object")
    if n_boot and n_boot > 0:
        raise NotImplementedError(
            "parametric-bootstrap ICC intervals are not implemented yet; "
            "call icc(result, n_boot=0) for the delta-method CI."
        )
    vc = result.variance_components
    alpha = float(alpha)

    # ---- three-level nested mixed() ---------------------------------------
    if getattr(result, "_cov_type", None) == "three-level-nested":
        levels = [k[len("var(_cons|") : -1] for k in vc if k.startswith("var(_cons|")]
        if component not in levels:
            raise KeyError(
                f"level {component!r} not found; available: {levels} "
                "(outer level first)"
            )
        s2 = [float(vc[f"var(_cons|{lv})"]) for lv in levels]
        total = sum(s2) + float(vc["var(Residual)"])
        rho = sum(s2[: levels.index(component) + 1]) / total
        warnings.warn(
            "icc(): standard errors are not available for three-level fits; "
            "se and the confidence interval are NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return ICCResult(float(rho), np.nan, np.nan, np.nan, alpha)

    key = f"var({component})"
    if key not in vc:
        raise KeyError(
            f"variance component {key!r} not found; " f"available: {list(vc)}"
        )
    x_random = list(getattr(result, "_x_random", []) or [])
    if x_random:
        raise MethodIncompatibility(
            "icc() is defined for random-intercept models only; with random "
            f"slopes on {x_random} the intraclass correlation depends on the "
            "covariate values.",
            recovery_hint="Refit with a random intercept only.",
            diagnostics={"x_random": x_random},
        )

    var_u = float(vc[key])
    family = getattr(result, "family", None)
    is_glmm = hasattr(result, "_cov_full")
    if is_glmm and family in ("binomial", "ordinal"):
        if getattr(result, "link", "logit") != "logit":
            raise MethodIncompatibility(
                "icc() latent-scale residual variance is defined here for the "
                "logit link only.",
                diagnostics={"link": getattr(result, "link", None)},
            )
        var_e = _LATENT_LOGIT_VAR
    elif "var(Residual)" in vc:
        var_e = float(vc["var(Residual)"])
    else:
        raise MethodIncompatibility(
            f"icc() is not defined for a {family!r} GLMM: there is no residual "
            "variance on the linear-predictor scale (Stata's estat icc is "
            "likewise unavailable after mepoisson / menbreg / meglm gamma).",
            recovery_hint="Use a Gaussian (sp.mixed) or logit (sp.melogit, "
            "sp.meologit) random-intercept model.",
            diagnostics={"family": family},
        )
    total = var_u + var_e
    if not (np.isfinite(total) and total > 0):
        raise MethodIncompatibility(
            "icc(): non-finite or non-positive total variance.",
            diagnostics={"var_u": var_u, "var_e": var_e},
        )
    rho = var_u / total
    c = _log_var_factor(getattr(result, "_cov_type", "unstructured"))

    # d rho / d log var_u = rho (1 - rho);  d rho / d log var_e = -rho (1 - rho)
    se = np.nan
    if is_glmm:
        cov_full = getattr(result, "_cov_full", None)
        if cov_full is not None:
            idx = len(result.fixed_effects) + (
                len(result.thresholds) if result.thresholds is not None else 0
            )
            g = np.zeros(cov_full.shape[0])
            g[idx] = c * rho * (1.0 - rho)
            if family == "gaussian":
                g[-1] = -rho * (1.0 - rho)  # packed log sigma2_e is last
            se = float(np.sqrt(g @ cov_full @ g))
    else:
        C = _mixed_vc_cov(result)
        if C is not None:
            g = np.array([c * rho * (1.0 - rho), -rho * (1.0 - rho)])
            se = float(np.sqrt(g @ C @ g))
    if not np.isfinite(se):
        warnings.warn(
            "icc(): the observed-information Hessian of the variance "
            "parameters is not positive definite (boundary fit?); se and "
            "the confidence interval are NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
    lo, hi = _logit_ci(rho, se, alpha)
    return ICCResult(float(rho), se, lo, hi, alpha)


__all__ = ["icc", "ICCResult"]
