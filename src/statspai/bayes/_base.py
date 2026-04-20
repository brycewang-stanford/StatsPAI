"""Shared types + PyMC lazy-import guard for ``statspai.bayes``.

PyMC is an **optional** dependency. Importing ``statspai.bayes`` never
imports PyMC; it is resolved at call time inside each estimator. If
PyMC is missing the estimator raises :class:`ImportError` with the
install recipe, leaving the rest of the package unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Optional-dependency guard
# ---------------------------------------------------------------------------

_PYMC_INSTALL_HINT = (
    "The statspai.bayes module requires PyMC + ArviZ. Install with:\n"
    "    pip install 'statspai[bayes]'\n"
    "or directly:\n"
    "    pip install pymc arviz"
)


def _require_pymc():
    """Import PyMC and ArviZ, or raise a clear ImportError."""
    try:
        import pymc as pm  # noqa: F401
        import arviz as az  # noqa: F401
    except ImportError as err:
        raise ImportError(_PYMC_INSTALL_HINT) from err
    return pm, az


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class BayesianCausalResult:
    """Summary of a Bayesian causal fit.

    A sibling of :class:`statspai.core.results.CausalResult` that
    speaks posterior (HDI, posterior probabilities, convergence
    diagnostics) instead of frequentist (CI, p-value).

    Attributes
    ----------
    method : str
        Human-readable method tag, e.g. ``"Bayesian DID (panel)"``.
    estimand : str
        Name of the causal estimand, e.g. ``"ATT"`` / ``"LATE"``.
    posterior_mean, posterior_median, posterior_sd : float
        Central tendency + dispersion of the posterior on the
        causal parameter.
    hdi_lower, hdi_upper : float
        Endpoints of the 95 % highest density interval.
    prob_positive : float
        Posterior probability that the estimand is > 0.
    prob_rope : float | None
        Posterior probability that the estimand lies in a
        user-supplied region of practical equivalence.
    rhat : float
        Gelman-Rubin potential scale reduction factor. Warn if
        > 1.01.
    ess : float
        Effective sample size (bulk).
    n_obs : int
        Sample size used in the fit.
    hdi_prob : float
        Nominal HDI coverage (default 0.95).
    trace : arviz.InferenceData | None
        Full posterior trace for downstream plotting / diagnostics.
    model_info : dict
        Misc. fit metadata (draws, tune, chains, priors, ...).
    """

    method: str
    estimand: str
    posterior_mean: float
    posterior_median: float
    posterior_sd: float
    hdi_lower: float
    hdi_upper: float
    prob_positive: float
    rhat: float
    ess: float
    n_obs: int
    hdi_prob: float = 0.95
    prob_rope: Optional[float] = None
    trace: Any = None  # arviz.InferenceData
    model_info: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Broom-style delegation (matches CausalResult.tidy / glance)
    # ------------------------------------------------------------------

    def tidy(self, conf_level: Optional[float] = None) -> pd.DataFrame:
        """Single-row DataFrame: term, estimate, std_error, HDI endpoints.

        Parameters
        ----------
        conf_level : float, optional
            Not used for Bayesian output (HDI has its own level set at
            fit time). Accepted for API parity with ``CausalResult.tidy``.
        """
        return pd.DataFrame([{
            'term': self.estimand.lower(),
            'estimate': self.posterior_mean,
            'std_error': self.posterior_sd,
            'statistic': (self.posterior_mean / self.posterior_sd
                          if self.posterior_sd > 0 else np.nan),
            'p_value': np.nan,  # not a frequentist concept
            'conf_low': self.hdi_lower,
            'conf_high': self.hdi_upper,
            'prob_positive': self.prob_positive,
            'hdi_prob': self.hdi_prob,
        }])

    def glance(self) -> pd.DataFrame:
        """Single-row DataFrame with fit-level diagnostics."""
        return pd.DataFrame([{
            'method': self.method,
            'nobs': self.n_obs,
            'rhat': self.rhat,
            'ess': self.ess,
            'chains': self.model_info.get('chains'),
            'draws': self.model_info.get('draws'),
            'tune': self.model_info.get('tune'),
            'hdi_prob': self.hdi_prob,
            'prob_positive': self.prob_positive,
        }])

    def summary(self) -> str:
        """Printable summary."""
        lines = [
            '=' * 70,
            f'  Method:      {self.method}',
            f'  Estimand:    {self.estimand}',
            f'  N obs:       {self.n_obs:,}',
            '-' * 70,
            f'  Posterior mean:   {self.posterior_mean:.4f}',
            f'  Posterior median: {self.posterior_median:.4f}',
            f'  Posterior SD:     {self.posterior_sd:.4f}',
            f'  {int(self.hdi_prob*100)}% HDI:          '
            f'[{self.hdi_lower:.4f}, {self.hdi_upper:.4f}]',
            f'  P({self.estimand} > 0):      {self.prob_positive:.3f}',
        ]
        if self.prob_rope is not None:
            lines.append(f'  P(|{self.estimand}| < rope): {self.prob_rope:.3f}')
        lines.extend([
            '-' * 70,
            'Convergence diagnostics',
            f'  R-hat:       {self.rhat:.4f}'
            + ('  [WARN > 1.01]' if self.rhat > 1.01 else ''),
            f'  ESS (bulk):  {self.ess:.0f}'
            + ('  [WARN: low ESS]' if self.ess < 400 else ''),
            f'  Chains:      {self.model_info.get("chains", "?")}',
            f'  Draws/chain: {self.model_info.get("draws", "?")}',
            '=' * 70,
        ])
        return '\n'.join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"<BayesianCausalResult {self.method}: "
                f"{self.estimand}={self.posterior_mean:.3f} "
                f"[HDI {self.hdi_lower:.3f}, {self.hdi_upper:.3f}] "
                f"rhat={self.rhat:.2f}>")


# ---------------------------------------------------------------------------
# Posterior summaries (helpers reused by did.py / rd.py)
# ---------------------------------------------------------------------------

def _sample_model(
    model,
    *,
    inference: str = 'nuts',
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_state: int = 42,
    progressbar: bool = False,
    advi_iterations: int = 20000,
):
    """Unified sampling entry-point for all Bayesian estimators.

    Parameters
    ----------
    model : ``pm.Model``
        An already-opened PyMC model (caller uses ``with pm.Model() as m:``).
    inference : {'nuts', 'advi'}, default 'nuts'
        - ``'nuts'`` : call :func:`pm.sample` (exact MCMC).
        - ``'advi'`` : call :func:`pm.fit` with mean-field ADVI, then
          draw ``draws`` samples from the fitted approximation. Much
          faster but mean-field assumes independent Gaussians so
          correlated posteriors look tighter than they are.
    draws, tune, chains, target_accept, random_state, progressbar :
        NUTS sampler controls.
    advi_iterations : int
        ADVI fit iterations.

    Returns
    -------
    arviz.InferenceData
        Directly usable by :func:`_summarise_posterior`.
    """
    pm, _ = _require_pymc()
    if inference not in ('nuts', 'advi'):
        raise ValueError(
            f"inference must be 'nuts' or 'advi'; got {inference!r}"
        )
    with model:
        if inference == 'advi':
            approx = pm.fit(
                n=advi_iterations,
                method='advi',
                random_seed=random_state,
                progressbar=progressbar,
            )
            trace = approx.sample(draws)
        else:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_state,
                progressbar=progressbar,
                return_inferencedata=True,
            )
    return trace


@dataclass
class BayesianHTEIVResult(BayesianCausalResult):
    """Extension of :class:`BayesianCausalResult` for heterogeneous-effect IV.

    Carries the average LATE (inherited) *plus* a table of CATE-slope
    posteriors — one row per effect modifier.
    """

    cate_slopes: pd.DataFrame = field(default_factory=pd.DataFrame)
    effect_modifiers: List[str] = field(default_factory=list)
    _modifier_means: Optional[np.ndarray] = None

    def predict_cate(self, values: Dict[str, float]) -> Dict[str, float]:
        """Posterior summary of CATE at specific modifier values.

        Parameters
        ----------
        values : dict[str, float]
            Map from modifier name → value. Missing modifiers default
            to the sample mean (i.e. zero contribution after centring).

        Returns
        -------
        dict
            Keys ``{'mean', 'median', 'sd', 'hdi_low', 'hdi_high',
            'prob_positive'}``.
        """
        if self.trace is None or self._modifier_means is None:
            raise RuntimeError(
                "predict_cate requires the posterior trace and "
                "modifier means to be present on the result."
            )
        _, az = _require_pymc()

        # Fetch posterior draws of tau_0 and tau_hte (shape (chains, draws) and (chains, draws, K))
        tau0 = self.trace.posterior['tau_0'].values.reshape(-1)
        tau_hte = self.trace.posterior['tau_hte'].values.reshape(
            -1, len(self.effect_modifiers)
        )

        # Build the modifier vector (deviation from training mean)
        m_vec = np.zeros(len(self.effect_modifiers))
        for i, name in enumerate(self.effect_modifiers):
            v = values.get(name, self._modifier_means[i])
            m_vec[i] = v - self._modifier_means[i]

        cate_samples = tau0 + tau_hte @ m_vec
        hdi = np.asarray(az.hdi(cate_samples, hdi_prob=self.hdi_prob)).ravel()
        return {
            'mean': float(np.mean(cate_samples)),
            'median': float(np.median(cate_samples)),
            'sd': float(np.std(cate_samples, ddof=1)),
            'hdi_low': float(hdi[0]),
            'hdi_high': float(hdi[1]),
            'prob_positive': float(np.mean(cate_samples > 0)),
        }


def _summarise_posterior(
    trace,
    var_name: str,
    hdi_prob: float = 0.95,
    rope: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """Compute posterior summary + convergence diagnostics for a scalar param.

    Parameters
    ----------
    trace : arviz.InferenceData
        Output of ``pm.sample``.
    var_name : str
        Name of the scalar parameter to summarise.
    hdi_prob : float
        HDI coverage.
    rope : (float, float), optional
        Region of practical equivalence; returns
        ``P(rope[0] < var_name < rope[1])`` as ``prob_rope``.
    """
    _, az = _require_pymc()
    posterior = trace.posterior[var_name].values.ravel()
    mean = float(np.mean(posterior))
    median = float(np.median(posterior))
    sd = float(np.std(posterior, ddof=1))
    hdi = az.hdi(posterior, hdi_prob=hdi_prob)
    # az.hdi may return a numpy array (older arviz) or DataArray
    hdi_arr = np.asarray(hdi).ravel()
    hdi_lower = float(hdi_arr[0])
    hdi_upper = float(hdi_arr[1])
    prob_positive = float(np.mean(posterior > 0))

    # Convergence diagnostics
    try:
        rhat = float(az.rhat(trace, var_names=[var_name])[var_name].values)
    except Exception:
        rhat = float('nan')
    try:
        ess = float(az.ess(trace, var_names=[var_name])[var_name].values)
    except Exception:
        ess = float('nan')

    out = {
        'posterior_mean': mean,
        'posterior_median': median,
        'posterior_sd': sd,
        'hdi_lower': hdi_lower,
        'hdi_upper': hdi_upper,
        'prob_positive': prob_positive,
        'rhat': rhat,
        'ess': ess,
    }
    if rope is not None:
        lo, hi = rope
        out['prob_rope'] = float(np.mean((posterior > lo) & (posterior < hi)))
    return out
