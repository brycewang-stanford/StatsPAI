"""Accelerated Failure Time (AFT) parametric survival models.

AFT models the **log survival time** directly as a linear function:

    log T_i = X_i β + σ ε_i

where the distribution of ε determines the family:

- **exponential**: ε ~ standard extreme-value (Gumbel)
- **weibull**: ε ~ standard extreme-value (Gumbel)
- **lognormal**: ε ~ N(0,1)
- **loglogistic**: ε ~ standard logistic

Estimation is by maximum likelihood with right-censoring.

References
----------
Kalbfleisch, J.D. & Prentice, R.L. (2002). *The Statistical Analysis
  of Failure Time Data*, 2nd ed. Wiley.
Klein, J.P. & Moeschberger, M.L. (2003). *Survival Analysis:
  Techniques for Censored and Truncated Data*, 2nd ed. Springer.
  [@kalbfleisch2002statistical]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility
from .models import _parse_formula

AFTFamily = Literal["exponential", "weibull", "lognormal", "loglogistic"]


@dataclass
class AFTResult(ResultProtocolMixin):
    beta: np.ndarray
    se: np.ndarray
    sigma: float
    var_names: List[str]
    family: AFTFamily
    log_likelihood: float
    aic: float
    bic: float
    n: int
    n_events: int
    se_log_sigma: float | None = None
    #: ``log_likelihood`` is of the log durations (Stata ``streg``); R
    #: ``survreg`` and ``sp.survreg`` report the likelihood of the
    #: durations, lower by the sum of ``log t`` over the events.
    log_likelihood_scale: str = "log time (Stata streg)"

    # ------------------------------------------------------------------
    # Agent-native accessors (consistent with EconometricResults /
    # sp.survreg). The regression coefficients live on the log-time scale;
    # ``log(sigma)`` is appended for the estimated-scale families (every
    # family except ``exponential``, whose scale is fixed at 1).
    # ------------------------------------------------------------------
    @property
    def params(self) -> pd.Series:
        names = list(self.var_names)
        vals = list(np.asarray(self.beta, dtype=float))
        if self.family != "exponential":
            names = names + ["log(sigma)"]
            vals = vals + [float(np.log(self.sigma))]
        return pd.Series(vals, index=names)

    @property
    def std_errors(self) -> pd.Series:
        names = list(self.var_names)
        vals = list(np.asarray(self.se, dtype=float))
        if self.family != "exponential":
            names = names + ["log(sigma)"]
            vals = vals + [
                (
                    float(self.se_log_sigma)
                    if self.se_log_sigma is not None
                    else float("nan")
                )
            ]
        return pd.Series(vals, index=names)

    @property
    def tvalues(self) -> pd.Series:
        return self.params / self.std_errors

    @property
    def pvalues(self) -> pd.Series:
        z = (self.params / self.std_errors).to_numpy(dtype=float)
        return pd.Series(
            2.0 * sp_stats.norm.sf(np.abs(z)),
            index=self.params.index,
        )

    def summary(self) -> str:
        lines = [
            f"AFT Model ({self.family})",
            "-" * 45,
            f"n = {self.n}, events = {self.n_events}",
            f"log σ = {np.log(self.sigma):.4f}  (σ = {self.sigma:.4f})",
            f"Log-Lik = {self.log_likelihood:.4f}",
            f"AIC = {self.aic:.2f}, BIC = {self.bic:.2f}",
            "",
            "Coefficients (on log-time scale):",
        ]
        for nm, b, s in zip(self.var_names, self.beta, self.se):
            t = b / s if s > 0 else np.nan
            p = 2 * sp_stats.norm.sf(abs(t))
            lines.append(
                f"  {nm:<15s}  {b: .4f}  (SE {s: .4f}, z {t: .3f}, p {p: .4f})"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def aft(
    formula: str,
    data: pd.DataFrame,
    family: AFTFamily = "weibull",
) -> AFTResult:
    """Fit an Accelerated Failure Time model by MLE.

    Parameters
    ----------
    formula : str
        ``"duration + event ~ x1 + x2"``. Durations must be positive and
        the event coded 0/1.
    family : {"exponential", "weibull", "lognormal", "loglogistic"}

    Returns
    -------
    AFTResult
        Coefficients and standard errors are those of :func:`sp.survreg`
        (the same model and maximum-likelihood fit). ``log_likelihood``
        (and the AIC / BIC built on it) is the likelihood of the log
        durations, as Stata ``streg`` prints it; ``sp.survreg``'s
        ``'Log-likelihood'`` is R ``survreg``'s likelihood of the
        durations, lower by the sum of ``log t`` over the events.
        ``log_likelihood_scale`` names the convention.

        .. versionchanged:: 1.41
            The fit is the exact MLE of ``sp.survreg``. Before, L-BFGS-B
            stopped about 1e-5 from it and the standard errors came from a
            finite-difference Hessian; zero durations were clamped to 1e-12
            and an event code of 2 entered the likelihood, both now refused.

    Examples
    --------
    Weibull AFT on simulated right-censored durations:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(7)
    >>> n = 300
    >>> x = rng.normal(size=n)
    >>> t = rng.exponential(np.exp(1.0 + 0.5 * x))
    >>> c = rng.exponential(np.exp(1.5), n)
    >>> df = pd.DataFrame({"dur": np.minimum(t, c),
    ...                    "event": (t <= c).astype(int), "x": x})
    >>> res = sp.aft("dur + event ~ x", data=df, family="weibull")
    >>> print(round(float(res.beta[1]), 2))  # truth 0.5 (log-time scale)
    0.52
    >>> print(res.summary().splitlines()[0])
    AFT Model (weibull)
    """
    lhs_str, covariates = _parse_formula(formula)
    lhs_parts = [s.strip() for s in lhs_str.split("+")]
    if len(lhs_parts) != 2:
        raise ValueError("formula LHS must be 'duration + event'")
    dur_col, event_col = lhs_parts

    if family not in ("exponential", "weibull", "lognormal", "loglogistic"):
        raise MethodIncompatibility(
            f"aft: unknown family {family!r}.",
            recovery_hint="Use 'exponential', 'weibull', 'lognormal' or "
            "'loglogistic'.",
            diagnostics={"family": family},
        )
    # The same model as sp.survreg, fitted by its exact maximum likelihood
    # (Newton-polished, observed-information SEs; Stata streg / R survreg
    # parity). Until 1.40.2 this ran L-BFGS-B at its default tolerance
    # (coefficients ~1e-5 from the MLE) with SEs from a finite-difference
    # Hessian of finite-difference gradients, clamped zero durations to
    # 1e-12 and read an event code of 2 into the likelihood; sp.survreg
    # refuses both.
    from .models import survreg

    fit = survreg(
        data=data, duration=dur_col, event=event_col, x=covariates, dist=family
    )
    fix_sigma = family == "exponential"
    k = len(covariates) + 1
    est = fit.params.to_numpy(dtype=float)
    se_all = fit.std_errors.to_numpy(dtype=float)
    beta = est[:k]
    se = se_all[:k]
    sigma = 1.0 if fix_sigma else float(np.exp(est[k]))
    se_log_sigma = None if fix_sigma else float(se_all[k])
    n = int(fit.data_info["nobs"])
    n_events = int(fit.data_info["n_events"])
    n_params = k + (0 if fix_sigma else 1)

    # Log-likelihood of the log durations, the scale Stata streg prints
    # (sp.survreg's 'Log-likelihood' is R survreg's, of the durations; the
    # two differ by the sum of log t over the events).
    ll = float(fit.model_info["ll_log_time"])
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + n_params * np.log(n)

    _result = AFTResult(
        beta=beta,
        se=se,
        sigma=sigma,
        var_names=["Intercept"] + covariates,
        family=family,
        log_likelihood=ll,
        aic=aic,
        bic=bic,
        n=n,
        n_events=n_events,
        se_log_sigma=se_log_sigma,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.survival.aft",
            params={"formula": formula, "family": family},
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
