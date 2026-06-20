"""Shared types + PyMC lazy-import guard for ``statspai.bayes``.

PyMC is an **optional** dependency. Importing ``statspai.bayes`` never
imports PyMC; it is resolved at call time inside each estimator. If
PyMC is missing the estimator raises :class:`ImportError` with the
install recipe, leaving the rest of the package unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability

# ---------------------------------------------------------------------------
# Optional-dependency guard
# ---------------------------------------------------------------------------

_PYMC_INSTALL_HINT = (
    "The statspai.bayes module requires PyMC + ArviZ. Install with:\n"
    "    pip install 'statspai[bayes]'\n"
    "or directly:\n"
    "    pip install pymc arviz"
)


# Shared clip constant for the probit-scale (``selection='normal'``)
# transform. Any site computing ``Φ^{-1}(a)`` — the fit-time
# polynomial powers, the post-hoc ATT/ATU integrator, and
# ``policy_effect`` — MUST use this constant so the three paths stay
# numerically consistent (cf. v0.9.12 round-C review HIGH on clip
# drift between fit and policy_effect). Tightening requires
# one-line change here; all callers pick it up automatically.
PROBIT_CLIP: float = 1e-6

_BAYES_ALTERNATIVES = [
    "sp.bayes_did",
    "sp.bayes_iv",
    "sp.bayes_rd",
    "sp.bayes_mte",
]


def _require_pymc() -> Tuple[Any, Any]:
    """Import PyMC and ArviZ, or raise a clear ImportError."""
    try:
        import pymc as pm  # noqa: F401
        import arviz as az  # noqa: F401
    except ImportError as err:
        raise ImportError(_PYMC_INSTALL_HINT) from err
    return pm, az


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise MethodIncompatibility(
            f"{name} must be a positive integer.",
            recovery_hint=f"Pass {name} as an integer greater than zero.",
            diagnostics={"parameter": name, "value": repr(value)},
            alternative_functions=_BAYES_ALTERNATIVES,
        )
    out = int(value)
    if out <= 0:
        raise MethodIncompatibility(
            f"{name} must be a positive integer.",
            recovery_hint=f"Increase {name} above zero.",
            diagnostics={"parameter": name, "value": out},
            alternative_functions=_BAYES_ALTERNATIVES,
        )
    return out


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise MethodIncompatibility(
            f"{name} must be a non-negative integer.",
            recovery_hint=f"Pass {name} as an integer greater than or equal to zero.",
            diagnostics={"parameter": name, "value": repr(value)},
            alternative_functions=_BAYES_ALTERNATIVES,
        )
    out = int(value)
    if out < 0:
        raise MethodIncompatibility(
            f"{name} must be a non-negative integer.",
            recovery_hint=f"Increase {name} to zero or above.",
            diagnostics={"parameter": name, "value": out},
            alternative_functions=_BAYES_ALTERNATIVES,
        )
    return out


def _open_unit_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MethodIncompatibility(
            f"{name} must be a finite float in (0, 1).",
            recovery_hint=f"Pass {name} as a probability-like float such as 0.9.",
            diagnostics={"parameter": name, "value": repr(value)},
            alternative_functions=_BAYES_ALTERNATIVES,
        )
    out = float(value)
    if not np.isfinite(out) or not 0.0 < out < 1.0:
        raise MethodIncompatibility(
            f"{name} must be a finite float in (0, 1).",
            recovery_hint=f"Choose a value strictly between 0 and 1 for {name}.",
            diagnostics={"parameter": name, "value": out},
            alternative_functions=_BAYES_ALTERNATIVES,
        )
    return out


def _normalise_terms(terms: Any, *, context: str) -> List[Any]:
    if isinstance(terms, str):
        return [terms]
    try:
        return list(terms)
    except TypeError as exc:
        raise MethodIncompatibility(
            f"{context} terms must be a string or an iterable of strings.",
            recovery_hint="Pass a single term string or a list such as "
            "['att', 'cohort:2019'].",
            diagnostics={"terms": repr(terms)},
            alternative_functions=_BAYES_ALTERNATIVES,
        ) from exc


def _validate_rope(
    rope: Optional[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    if rope is None:
        return None
    try:
        lo, hi = rope
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "rope must be a two-value tuple (lower, upper).",
            recovery_hint="Pass rope=(lower, upper) with finite numeric bounds.",
            diagnostics={"rope": repr(rope)},
            alternative_functions=_BAYES_ALTERNATIVES,
        ) from exc
    if not isinstance(lo, Real) or not isinstance(hi, Real):
        raise MethodIncompatibility(
            "rope bounds must be numeric.",
            recovery_hint="Pass finite numeric lower and upper bounds.",
            diagnostics={"rope": repr(rope)},
            alternative_functions=_BAYES_ALTERNATIVES,
        )
    lo_f = float(lo)
    hi_f = float(hi)
    if not np.isfinite([lo_f, hi_f]).all() or lo_f >= hi_f:
        raise MethodIncompatibility(
            "rope bounds must be finite and ordered as lower < upper.",
            recovery_hint="Use a finite interval such as rope=(-0.1, 0.1).",
            diagnostics={"lower": lo_f, "upper": hi_f},
            alternative_functions=_BAYES_ALTERNATIVES,
        )
    return lo_f, hi_f


def _az_hdi_compat(samples: Any, hdi_prob: float = 0.95) -> np.ndarray:
    """Call ``arviz.hdi`` with the correct kwarg regardless of version.

    arviz < 0.18 accepts ``hdi_prob=...``; arviz ≥ 0.18 renamed it to
    ``prob=...``. Everything else downstream expects a length-2 numpy
    array (``[lower, upper]``), so we normalise the output too.

    Code-review history: the v0.9.12 round-C review flagged the
    direct ``az.hdi(..., hdi_prob=...)`` calls as a time-bomb for
    arviz ≥ 0.18 upgrades. This shim routes all calls through one
    place so a future arviz rename only touches this function.
    """
    _, az = _require_pymc()
    try:
        return np.asarray(az.hdi(samples, hdi_prob=hdi_prob)).ravel()
    except TypeError:
        return np.asarray(az.hdi(samples, prob=hdi_prob)).ravel()


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

    Examples
    --------
    >>> import statspai as sp
    >>> # Returned by every scalar-estimand Bayesian estimator, e.g.
    >>> # bayes_rd / bayes_iv (requires the `bayes` extra:
    >>> #   pip install 'statspai[bayes]').
    >>> res = sp.bayes_rd(df, y='y', running='x', cutoff=0.0)  # doctest: +SKIP
    >>> isinstance(res, sp.BayesianCausalResult)  # doctest: +SKIP
    >>> res.posterior_mean, res.hdi_lower, res.hdi_upper  # doctest: +SKIP
    >>> res.prob_positive   # posterior P(estimand > 0)  # doctest: +SKIP
    >>> res.rhat, res.ess   # convergence diagnostics  # doctest: +SKIP
    >>> res.tidy()  # doctest: +SKIP
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
        return pd.DataFrame(
            [
                {
                    "term": self.estimand.lower(),
                    "estimate": self.posterior_mean,
                    "std_error": self.posterior_sd,
                    "statistic": (
                        self.posterior_mean / self.posterior_sd
                        if self.posterior_sd > 0
                        else np.nan
                    ),
                    "p_value": np.nan,  # not a frequentist concept
                    "conf_low": self.hdi_lower,
                    "conf_high": self.hdi_upper,
                    "prob_positive": self.prob_positive,
                    "hdi_prob": self.hdi_prob,
                }
            ]
        )

    def glance(self) -> pd.DataFrame:
        """Single-row DataFrame with fit-level diagnostics."""
        return pd.DataFrame(
            [
                {
                    "method": self.method,
                    "nobs": self.n_obs,
                    "rhat": self.rhat,
                    "ess": self.ess,
                    "chains": self.model_info.get("chains"),
                    "draws": self.model_info.get("draws"),
                    "tune": self.model_info.get("tune"),
                    "hdi_prob": self.hdi_prob,
                    "prob_positive": self.prob_positive,
                }
            ]
        )

    def summary(self) -> str:
        """Printable summary."""
        lines = [
            "=" * 70,
            f"  Method:      {self.method}",
            f"  Estimand:    {self.estimand}",
            f"  N obs:       {self.n_obs:,}",
            "-" * 70,
            f"  Posterior mean:   {self.posterior_mean:.4f}",
            f"  Posterior median: {self.posterior_median:.4f}",
            f"  Posterior SD:     {self.posterior_sd:.4f}",
            f"  {int(self.hdi_prob*100)}% HDI:          "
            f"[{self.hdi_lower:.4f}, {self.hdi_upper:.4f}]",
            f"  P({self.estimand} > 0):      {self.prob_positive:.3f}",
        ]
        if self.prob_rope is not None:
            lines.append(f"  P(|{self.estimand}| < rope): {self.prob_rope:.3f}")
        # R-hat / ESS formatting has to distinguish three cases:
        #   1. NUTS with rhat>1.01 -> loud warning
        #   2. NUTS with healthy rhat -> quiet
        #   3. ADVI (rhat is NaN because 1 chain) -> different warning
        #      that tells the user convergence is undiagnosable, not
        #      that the model failed.
        inference_mode = self.model_info.get("inference", "nuts")
        # Variational backends (ADVI, Pathfinder) produce a single
        # drawn chain — R-hat is meaningless. Exact samplers (NUTS,
        # SMC) have meaningful R-hat.
        is_variational = inference_mode in ("advi", "pathfinder")
        if is_variational or np.isnan(self.rhat):
            label = {
                "advi": "mean-field ADVI",
                "pathfinder": "full-rank ADVI (Pathfinder stand-in)",
            }.get(inference_mode, "variational")
            rhat_line = (
                f"  R-hat:       n/a  [{label} — convergence is "
                "not measurable via R-hat; use NUTS or SMC for "
                "calibrated uncertainty]"
            )
        elif self.rhat > 1.01:
            rhat_line = f"  R-hat:       {self.rhat:.4f}  [WARN > 1.01]"
        else:
            rhat_line = f"  R-hat:       {self.rhat:.4f}"

        ess_line = f"  ESS (bulk):  {self.ess:.0f}"
        if not is_variational and np.isfinite(self.ess) and self.ess < 400:
            ess_line += "  [WARN: low ESS]"

        # Effective chain count is 1 under variational backends.
        requested_chains = self.model_info.get("chains", "?")
        if is_variational:
            chains_line = (
                f"  Chains:      1  ({inference_mode}; requested "
                f"{requested_chains} ignored)"
            )
        else:
            chains_line = f"  Chains:      {requested_chains}"

        lines.extend(
            [
                "-" * 70,
                "Convergence diagnostics",
                rhat_line,
                ess_line,
                chains_line,
                f'  Draws/chain: {self.model_info.get("draws", "?")}',
                f"  Inference:   {inference_mode}",
                "=" * 70,
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _jsonable(value: Any) -> Any:
        """Coerce posterior payload values to strict JSON-safe primitives."""
        if value is None:
            return None
        if isinstance(value, (bool, str)):
            return value
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            v = float(value)
            return v if np.isfinite(v) else None
        if isinstance(value, dict):
            return {str(k): BayesianCausalResult._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [BayesianCausalResult._jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return BayesianCausalResult._jsonable(value.tolist())
        if isinstance(value, pd.Series):
            return BayesianCausalResult._jsonable(value.to_dict())
        if isinstance(value, pd.DataFrame):
            return BayesianCausalResult._jsonable(value.to_dict(orient="records"))
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return str(value)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe Bayesian-result payload without posterior trace.

        The full ArviZ trace can be large and may carry backend-specific
        objects. This method serializes the stable decision surface: posterior
        point/interval summaries, convergence diagnostics, tidy/glance records,
        and JSON-safe fit metadata.
        """
        tidy_records = self.tidy().to_dict(orient="records")
        glance_records = self.glance().to_dict(orient="records")
        return {
            "kind": "bayesian_causal_result",
            "method": self.method,
            "estimand": self.estimand,
            "n_obs": self._jsonable(self.n_obs),
            "posterior": {
                "mean": self._jsonable(self.posterior_mean),
                "median": self._jsonable(self.posterior_median),
                "sd": self._jsonable(self.posterior_sd),
                "hdi": {
                    "level": self._jsonable(self.hdi_prob),
                    "lower": self._jsonable(self.hdi_lower),
                    "upper": self._jsonable(self.hdi_upper),
                },
                "prob_positive": self._jsonable(self.prob_positive),
                "prob_rope": self._jsonable(self.prob_rope),
            },
            "diagnostics": {
                "rhat": self._jsonable(self.rhat),
                "ess": self._jsonable(self.ess),
                "inference": self._jsonable(self.model_info.get("inference", "nuts")),
                "chains": self._jsonable(self.model_info.get("chains")),
                "draws": self._jsonable(self.model_info.get("draws")),
                "tune": self._jsonable(self.model_info.get("tune")),
            },
            "model_info": self._jsonable(self.model_info),
            "tidy": self._jsonable(tidy_records),
            "glance": self._jsonable(glance_records),
        }

    def to_agent_summary(self) -> Dict[str, Any]:
        """Return a compact nested summary for LLM/tool-loop decisions."""
        payload = self.to_dict()
        rhat = payload["diagnostics"]["rhat"]
        ess = payload["diagnostics"]["ess"]
        warnings: List[str] = []
        if rhat is None:
            warnings.append("R-hat unavailable; convergence cannot be checked.")
        elif rhat > 1.01:
            warnings.append("R-hat exceeds 1.01; inspect convergence.")
        if ess is not None and ess < 400:
            warnings.append("Bulk ESS is below 400; posterior may be noisy.")
        return {
            "kind": "bayesian_causal_agent_summary",
            "method": payload["method"],
            "estimand": payload["estimand"],
            "n_obs": payload["n_obs"],
            "posterior": payload["posterior"],
            "diagnostics": payload["diagnostics"],
            "warnings": warnings,
            "model_info": payload["model_info"],
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<BayesianCausalResult {self.method}: "
            f"{self.estimand}={self.posterior_mean:.3f} "
            f"[HDI {self.hdi_lower:.3f}, {self.hdi_upper:.3f}] "
            f"rhat={self.rhat:.2f}>"
        )


# ---------------------------------------------------------------------------
# Posterior summaries (helpers reused by did.py / rd.py)
# ---------------------------------------------------------------------------


def _sample_model(
    model: Any,
    *,
    inference: str = "nuts",
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_state: int = 42,
    progressbar: bool = False,
    advi_iterations: int = 20000,
) -> Any:
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
    valid = ("nuts", "advi", "pathfinder", "smc")
    if not isinstance(inference, str) or inference not in valid:
        raise MethodIncompatibility(
            f"inference must be one of {valid}; got {inference!r}.",
            recovery_hint="Use inference='nuts' for calibrated MCMC or "
            "inference='advi' for a faster variational approximation.",
            diagnostics={"parameter": "inference", "valid": valid},
            alternative_functions=_BAYES_ALTERNATIVES,
        )
    draws = _positive_int(draws, "draws")
    tune = _nonnegative_int(tune, "tune")
    chains = _positive_int(chains, "chains")
    advi_iterations = _positive_int(advi_iterations, "advi_iterations")
    target_accept = _open_unit_float(target_accept, "target_accept")

    pm, _ = _require_pymc()
    with model:
        if inference == "advi":
            approx = pm.fit(
                n=advi_iterations,
                method="advi",
                random_seed=random_state,
                progressbar=progressbar,
            )
            trace = approx.sample(draws)
            # Record that the effective chain count is 1 regardless of
            # what the caller requested. Without this the glance()
            # output misleads — posterior has 1 chain but user asked
            # for 4.
            trace.attrs["actual_chains"] = 1
            trace.attrs["inference"] = "advi"
        elif inference == "pathfinder":
            # PyMC 5.x's stable "smarter-than-mean-field" VI entry-point
            # is full-rank ADVI — it captures pairwise covariance between
            # parameters, which mean-field ADVI misses. When PyMC's
            # `pmx.fit` stabilises we will switch to true Pathfinder;
            # until then full-rank ADVI is the same spirit.
            approx = pm.fit(
                n=advi_iterations,
                method="fullrank_advi",
                random_seed=random_state,
                progressbar=progressbar,
            )
            trace = approx.sample(draws)
            trace.attrs["actual_chains"] = 1
            trace.attrs["inference"] = "pathfinder"
        elif inference == "smc":
            # Sequential Monte Carlo — exact sampler that handles
            # multimodal posteriors well; slower than NUTS on unimodal
            # problems but a clean fallback when NUTS diverges.
            trace = pm.sample_smc(
                draws=draws,
                chains=chains,
                random_seed=random_state,
                progressbar=progressbar,
            )
            trace.attrs["actual_chains"] = chains
            trace.attrs["inference"] = "smc"
        else:  # 'nuts'
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_state,
                progressbar=progressbar,
                return_inferencedata=True,
            )
            trace.attrs["actual_chains"] = chains
            trace.attrs["inference"] = "nuts"
    return trace


def _tidy_row_from_summary(
    term_label: str,
    summary: Dict[str, float],
    hdi_prob: float,
) -> Dict[str, Any]:
    """Build a tidy-schema row from a per-subgroup summary dict.

    Both :class:`BayesianDIDResult` and :class:`BayesianIVResult` use
    the same multi-row tidy schema so downstream ``pd.concat`` /
    ``modelsummary`` pipelines see a rectangular table even when
    subgroups mix (e.g. one DID result with cohort splits alongside
    one IV result with per-instrument LATEs).
    """
    est = summary.get("posterior_mean", float("nan"))
    sd = summary.get("posterior_sd", float("nan"))
    lo = summary.get("hdi_lower", float("nan"))
    hi = summary.get("hdi_upper", float("nan"))
    pp = summary.get("prob_positive", float("nan"))
    stat = (est / sd) if (sd is not None and np.isfinite(sd) and sd > 0) else np.nan
    return {
        "term": term_label,
        "estimate": est,
        "std_error": sd,
        "statistic": stat,
        "p_value": np.nan,
        "conf_low": lo,
        "conf_high": hi,
        "prob_positive": pp,
        "hdi_prob": hdi_prob,
    }


@dataclass
class BayesianDIDResult(BayesianCausalResult):
    """Bayesian DID result with optional per-cohort ATT posteriors.

    Extends :class:`BayesianCausalResult` with a
    ``cohort_summaries`` mapping ``cohort-label -> summary-dict``
    populated by :func:`bayes_did` when called with ``cohort=...``.
    When the dict is empty (default), the result behaves exactly
    like :class:`BayesianCausalResult` and ``tidy()`` produces the
    same single-row DataFrame as before.

    The ``cohort_labels`` field preserves the user's original cohort
    values in the order the model sampled them; iteration order is
    deterministic and matches the posterior variable axis.

    Examples
    --------
    >>> import statspai as sp
    >>> # Returned by bayes_did (requires the `bayes` extra:
    >>> #   pip install 'statspai[bayes]'). Pass cohort=... to populate
    >>> # the per-cohort posteriors.
    >>> res = sp.bayes_did(df, y='y', treat='treat', post='post',
    ...                    cohort='first_treat')  # doctest: +SKIP
    >>> res.tidy()                       # pooled ATT row  # doctest: +SKIP
    >>> res.tidy(terms='per_cohort')     # one row per cohort  # doctest: +SKIP
    >>> res.cohort_summaries  # doctest: +SKIP
    """

    cohort_summaries: Dict[str, Dict[str, float]] = field(default_factory=dict)
    cohort_labels: List[Any] = field(default_factory=list)

    def tidy(
        self,
        conf_level: Optional[float] = None,
        terms: Any = None,
    ) -> pd.DataFrame:
        """Tidy summary with optional per-cohort breakdown.

        Parameters
        ----------
        conf_level : float, optional
            Unused for Bayesian output; accepted for API parity with
            :class:`CausalResult.tidy`.
        terms : None | str | sequence of str, default ``None``
            - ``None`` / ``'att'`` — single average-ATT row (v0.9.14
              behaviour).
            - ``'per_cohort'`` — one row per cohort (requires
              ``cohort_summaries`` to be populated; otherwise raises).
            - list like ``['att', 'cohort:2019', 'cohort:2020']`` —
              explicit selection. Cohort labels use the prefix
              ``'cohort:'`` followed by the user's original cohort
              value coerced to string. Unknown cohort labels raise.
        """
        if terms is None:
            return super().tidy(conf_level=conf_level)

        if isinstance(terms, str):
            if terms == "att":
                return super().tidy(conf_level=conf_level)
            if terms == "per_cohort":
                if not self.cohort_summaries:
                    raise MethodIncompatibility(
                        "terms='per_cohort' requires a fit with "
                        "cohort=... populating cohort_summaries.",
                        recovery_hint="Refit sp.bayes_did(..., cohort=...) or "
                        "request terms='att'.",
                        diagnostics={"terms": terms, "has_cohorts": False},
                        alternative_functions=["sp.bayes_did"],
                    )
                term_list = [f"cohort:{c}" for c in self.cohort_labels]
            else:
                term_list = [terms]
        else:
            term_list = _normalise_terms(terms, context="Bayesian DID tidy")
            if term_list == [self.estimand.lower()]:
                return super().tidy(conf_level=conf_level)

        rows: List[Dict[str, Any]] = []
        # Back-compat label for the average ATT so downstream concat
        # pipelines can freely mix 'att' with 'cohort:*' rows.
        known_avg = {"att", self.estimand.lower()}
        cohort_keys = {f"cohort:{c}" for c in self.cohort_labels}
        for term in term_list:
            if term in known_avg:
                rows.append(
                    _tidy_row_from_summary(
                        self.estimand.lower(),
                        {
                            "posterior_mean": self.posterior_mean,
                            "posterior_sd": self.posterior_sd,
                            "hdi_lower": self.hdi_lower,
                            "hdi_upper": self.hdi_upper,
                            "prob_positive": self.prob_positive,
                        },
                        self.hdi_prob,
                    )
                )
            elif term in cohort_keys:
                raw_label = term[len("cohort:") :]
                # Map back to the user's original dict key: we stored
                # it as str(label) in cohort_summaries.
                summary = self.cohort_summaries[raw_label]
                rows.append(
                    _tidy_row_from_summary(
                        term,
                        summary,
                        self.hdi_prob,
                    )
                )
            else:
                raise MethodIncompatibility(
                    f"Unknown term {term!r}. Valid options: "
                    f"'att' (average), 'per_cohort', or a cohort label "
                    f"in {sorted(cohort_keys)}.",
                    recovery_hint="Request terms='att', terms='per_cohort', "
                    "or one of the returned cohort labels.",
                    diagnostics={
                        "term": term,
                        "valid_terms": ["att", "per_cohort", *sorted(cohort_keys)],
                    },
                    alternative_functions=["sp.bayes_did"],
                )
        return pd.DataFrame(rows)


@dataclass
class BayesianIVResult(BayesianCausalResult):
    """Bayesian IV result with optional per-instrument LATE posteriors.

    Extends :class:`BayesianCausalResult` with an
    ``instrument_summaries`` mapping ``instrument-name -> summary-dict``
    populated by :func:`bayes_iv` when called with
    ``per_instrument=True``. When the dict is empty (default),
    behaviour matches v0.9.15 exactly.

    Examples
    --------
    >>> import statspai as sp
    >>> # Returned by bayes_iv (requires the `bayes` extra:
    >>> #   pip install 'statspai[bayes]'). Pass per_instrument=True with
    >>> # multiple instruments to populate the per-Z posteriors.
    >>> res = sp.bayes_iv(df, y='y', treat='d',
    ...                   instrument=['z1', 'z2'],
    ...                   per_instrument=True)  # doctest: +SKIP
    >>> res.tidy()                          # pooled LATE row  # doctest: +SKIP
    >>> res.tidy(terms='per_instrument')  # doctest: +SKIP
    >>> res.instrument_summaries  # doctest: +SKIP
    """

    instrument_summaries: Dict[str, Dict[str, float]] = field(default_factory=dict)
    instrument_labels: List[str] = field(default_factory=list)

    def tidy(
        self,
        conf_level: Optional[float] = None,
        terms: Any = None,
    ) -> pd.DataFrame:
        """Tidy summary with optional per-instrument breakdown.

        Parameters
        ----------
        terms : None | str | sequence of str, default ``None``
            - ``None`` / ``'late'`` — single pooled-LATE row.
            - ``'per_instrument'`` — one row per instrument (requires
              a fit with ``per_instrument=True``).
            - list like ``['late', 'instrument:z1', 'instrument:z2']`` —
              explicit selection. Unknown labels raise.
        """
        if terms is None:
            return super().tidy(conf_level=conf_level)

        if isinstance(terms, str):
            if terms == "late":
                return super().tidy(conf_level=conf_level)
            if terms == "per_instrument":
                if not self.instrument_summaries:
                    raise MethodIncompatibility(
                        "terms='per_instrument' requires a fit with "
                        "per_instrument=True populating "
                        "instrument_summaries.",
                        recovery_hint="Refit sp.bayes_iv(..., "
                        "per_instrument=True) or request terms='late'.",
                        diagnostics={
                            "terms": terms,
                            "has_instruments": False,
                        },
                        alternative_functions=["sp.bayes_iv"],
                    )
                term_list = [f"instrument:{z}" for z in self.instrument_labels]
            else:
                term_list = [terms]
        else:
            term_list = _normalise_terms(terms, context="Bayesian IV tidy")
            if term_list == [self.estimand.lower()]:
                return super().tidy(conf_level=conf_level)

        rows: List[Dict[str, Any]] = []
        known_avg = {"late", self.estimand.lower()}
        instr_keys = {f"instrument:{z}" for z in self.instrument_labels}
        for term in term_list:
            if term in known_avg:
                rows.append(
                    _tidy_row_from_summary(
                        self.estimand.lower(),
                        {
                            "posterior_mean": self.posterior_mean,
                            "posterior_sd": self.posterior_sd,
                            "hdi_lower": self.hdi_lower,
                            "hdi_upper": self.hdi_upper,
                            "prob_positive": self.prob_positive,
                        },
                        self.hdi_prob,
                    )
                )
            elif term in instr_keys:
                raw_label = term[len("instrument:") :]
                summary = self.instrument_summaries[raw_label]
                rows.append(
                    _tidy_row_from_summary(
                        term,
                        summary,
                        self.hdi_prob,
                    )
                )
            else:
                raise MethodIncompatibility(
                    f"Unknown term {term!r}. Valid options: "
                    f"'late' (pooled), 'per_instrument', or an "
                    f"instrument label in {sorted(instr_keys)}.",
                    recovery_hint="Request terms='late', "
                    "terms='per_instrument', or one of the returned "
                    "instrument labels.",
                    diagnostics={
                        "term": term,
                        "valid_terms": [
                            "late",
                            "per_instrument",
                            *sorted(instr_keys),
                        ],
                    },
                    alternative_functions=["sp.bayes_iv"],
                )
        return pd.DataFrame(rows)


@dataclass
class BayesianHTEIVResult(BayesianCausalResult):
    """Extension of :class:`BayesianCausalResult` for heterogeneous-effect IV.

    Carries the average LATE (inherited) *plus* a table of CATE-slope
    posteriors — one row per effect modifier.

    Examples
    --------
    >>> import statspai as sp
    >>> # Returned by bayes_hte_iv (requires the `bayes` extra:
    >>> #   pip install 'statspai[bayes]').
    >>> res = sp.bayes_hte_iv(df, y='y', treat='d', instrument='z',
    ...                       effect_modifiers=['age'])  # doctest: +SKIP
    >>> res.posterior_mean  # doctest: +SKIP
    >>> res.cate_slopes  # doctest: +SKIP
    >>> res.predict_cate({'age': 40})  # doctest: +SKIP
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
            raise MethodIncompatibility(
                "predict_cate requires the posterior trace and "
                "modifier means to be present on the result.",
                recovery_hint="Call predict_cate() on a fitted "
                "BayesianHTEIVResult rather than a serialized summary.",
                diagnostics={
                    "has_trace": self.trace is not None,
                    "has_modifier_means": self._modifier_means is not None,
                },
                alternative_functions=["sp.bayes_hte_iv"],
            )
        if not hasattr(values, "get"):
            raise MethodIncompatibility(
                "predict_cate values must be a mapping from modifier to value.",
                recovery_hint="Pass a dict such as {'age': 40}.",
                diagnostics={"type": type(values).__name__},
                alternative_functions=["sp.bayes_hte_iv"],
            )

        # Fetch posterior draws of tau_0 and tau_hte.
        tau0 = self.trace.posterior["tau_0"].values.reshape(-1)
        tau_hte = self.trace.posterior["tau_hte"].values.reshape(
            -1, len(self.effect_modifiers)
        )

        # Build the modifier vector (deviation from training mean)
        m_vec = np.zeros(len(self.effect_modifiers))
        for i, name in enumerate(self.effect_modifiers):
            v = values.get(name, self._modifier_means[i])
            if not isinstance(v, Real) or not np.isfinite(float(v)):
                raise MethodIncompatibility(
                    f"Modifier value for {name!r} must be finite numeric.",
                    recovery_hint="Pass finite numeric values for all requested "
                    "effect modifiers.",
                    diagnostics={"modifier": name, "value": repr(v)},
                    alternative_functions=["sp.bayes_hte_iv"],
                )
            m_vec[i] = float(v) - self._modifier_means[i]

        cate_samples = tau0 + tau_hte @ m_vec
        hdi = _az_hdi_compat(cate_samples, hdi_prob=self.hdi_prob)
        return {
            "mean": float(np.mean(cate_samples)),
            "median": float(np.median(cate_samples)),
            "sd": float(np.std(cate_samples, ddof=1)),
            "hdi_low": float(hdi[0]),
            "hdi_high": float(hdi[1]),
            "prob_positive": float(np.mean(cate_samples > 0)),
        }


@dataclass
class BayesianMTEResult(BayesianCausalResult):
    """Bayesian Marginal Treatment Effect result.

    Carries (in addition to the inherited average MTE summary) a
    full posterior over the MTE curve ``tau(u)`` on a user-specified
    grid, plus integrated summaries over the treated / untreated /
    average populations (ATT, ATU, ATE).

    Examples
    --------
    >>> import statspai as sp
    >>> # Returned by bayes_mte (requires the `bayes` extra:
    >>> #   pip install 'statspai[bayes]').
    >>> res = sp.bayes_mte(df, y='y', treat='d',
    ...                    instrument='z')  # doctest: +SKIP
    >>> res.ate, res.att, res.atu  # doctest: +SKIP
    >>> res.mte_curve  # doctest: +SKIP
    >>> res.tidy(terms=['ate', 'att', 'atu'])  # doctest: +SKIP
    >>> # Policy-relevant effect with a ready-made weight builder:
    >>> res.policy_effect(sp.policy_weight_ate())  # doctest: +SKIP
    """

    mte_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    u_grid: Optional[np.ndarray] = None
    ate: float = float("nan")
    att: float = float("nan")
    atu: float = float("nan")
    # v0.9.12: `'uniform'` (default) fits MTE polynomial in U_D on
    # [0,1]; `'normal'` fits in V = Φ^{-1}(U_D) on ℝ. The result
    # remembers this so `policy_effect` can transform the abscissa
    # before integrating against the user's weight_fn.
    selection: str = "uniform"
    # v0.9.13: ATT / ATU uncertainty triplets. Default NaN so
    # pre-v0.9.13 serialised results back-compat. ``posterior_sd`` on
    # the parent already covers ATE uncertainty (ATE is the primary
    # estimand) so there is no ``ate_sd`` field.
    att_sd: float = float("nan")
    att_hdi_lower: float = float("nan")
    att_hdi_upper: float = float("nan")
    atu_sd: float = float("nan")
    atu_hdi_lower: float = float("nan")
    atu_hdi_upper: float = float("nan")
    # v0.9.15: posterior P(ATT > 0) / P(ATU > 0). NaN by default so
    # results deserialised from earlier snapshots do not break; when
    # v0.9.15+ bayes_mte fits, these scalars are populated from the
    # per-draw ATT / ATU posterior and exposed via
    # ``tidy(terms=['att', 'atu'])``.
    att_prob_positive: float = float("nan")
    atu_prob_positive: float = float("nan")

    def summary(self) -> str:
        """Printable summary with ATT/ATU uncertainty appended.

        Extends ``BayesianCausalResult.summary`` with a block printing
        ``ATT`` / ``ATU`` posterior mean, SD and HDI when the SD
        fields are finite. Silently skipped (parent summary only) when
        either side of the population is empty or the result was
        deserialised from a pre-v0.9.13 snapshot (fields default NaN).
        """
        base = super().summary()
        extra: list[str] = []
        pct = int(self.hdi_prob * 100)
        if np.isfinite(self.att_sd):
            extra.append(
                f"  ATT: {self.att:.4f} (sd {self.att_sd:.4f}, "
                f"{pct}% HDI [{self.att_hdi_lower:.4f}, "
                f"{self.att_hdi_upper:.4f}])"
            )
        if np.isfinite(self.atu_sd):
            extra.append(
                f"  ATU: {self.atu:.4f} (sd {self.atu_sd:.4f}, "
                f"{pct}% HDI [{self.atu_hdi_lower:.4f}, "
                f"{self.atu_hdi_upper:.4f}])"
            )
        if not extra:
            return base
        block = "\n".join(["-" * 70, "Population-integrated effects", *extra])
        closing = "=" * 70
        if base.endswith(closing):
            return base[: -len(closing)] + block + "\n" + closing
        return base + "\n" + block

    def tidy(
        self,
        conf_level: Optional[float] = None,
        terms: Any = None,
    ) -> pd.DataFrame:
        """Broom-style tidy summary with optional multi-term output.

        Extends :meth:`BayesianCausalResult.tidy` so a single fit can
        emit a long-format DataFrame for ATE / ATT / ATU in one call,
        which is what downstream meta-analysis pipelines (pd.concat
        + modelsummary, gt, etc.) expect.

        Parameters
        ----------
        conf_level : float, optional
            Unused for Bayesian output; accepted for API parity with
            :class:`CausalResult.tidy`.
        terms : None | str | sequence of str, default ``None``
            Which term(s) to include:

            - ``None`` or ``'ate'`` — single ATE row (back-compat
              with v0.9.14 default).
            - ``'att'`` / ``'atu'`` — single row of that term.
            - list like ``['ate', 'att', 'atu']`` — multi-row.

            Unknown names raise :class:`ValueError`. When an ATT /
            ATU term is requested but the corresponding SD field is
            NaN (empty subpopulation, or result deserialised from
            pre-v0.9.13), the row is still emitted with NaN
            uncertainty columns — the schema stays rectangular so
            downstream concat doesn't misalign columns.
        """
        # Default path stays byte-identical to the parent v0.9.14
        # implementation.
        if terms is None:
            return super().tidy(conf_level=conf_level)

        # Normalise to a list of requested term labels.
        if isinstance(terms, str):
            terms_list = [terms]
        else:
            terms_list = _normalise_terms(terms, context="Bayesian MTE tidy")

        known = {"ate", "att", "atu"}
        unknown = [t for t in terms_list if t not in known]
        if unknown:
            raise MethodIncompatibility(
                f"Unknown term(s) {unknown}; valid options are " f"{sorted(known)}.",
                recovery_hint="Request one or more of 'ate', 'att', or 'atu'.",
                diagnostics={"unknown_terms": unknown, "valid_terms": sorted(known)},
                alternative_functions=["sp.bayes_mte"],
            )

        def _row(term: str) -> Dict[str, Any]:
            # Round-B review-H1 fix: for ``term == 'ate'`` we use the
            # same label the parent default path uses (``estimand.lower()``
            # = ``'ate (integrated mte)'``) so ``tidy(terms='ate')`` and
            # plain ``tidy()`` produce *byte-identical* rows for the
            # ATE term. Mixing short/long labels inside the same concat
            # pipeline was the original divergence concern.
            if term == "ate":
                term_label = self.estimand.lower()
                est = self.posterior_mean
                sd = self.posterior_sd
                lo = self.hdi_lower
                hi = self.hdi_upper
                pp = self.prob_positive
            elif term == "att":
                term_label = "att"
                est = self.att
                sd = self.att_sd
                lo = self.att_hdi_lower
                hi = self.att_hdi_upper
                pp = self.att_prob_positive
            else:  # 'atu'
                term_label = "atu"
                est = self.atu
                sd = self.atu_sd
                lo = self.atu_hdi_lower
                hi = self.atu_hdi_upper
                pp = self.atu_prob_positive
            stat = (
                (est / sd)
                if (sd is not None and np.isfinite(sd) and sd > 0)
                else np.nan
            )
            return {
                "term": term_label,
                "estimate": est,
                "std_error": sd,
                "statistic": stat,
                "p_value": np.nan,
                "conf_low": lo,
                "conf_high": hi,
                "prob_positive": pp,
                "hdi_prob": self.hdi_prob,
            }

        return pd.DataFrame([_row(t) for t in terms_list])

    def policy_effect(
        self,
        weight_fn: Any,
        label: str = "policy",
        rope: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """Posterior summary of a policy-relevant treatment effect.

        Computes ``E[w(U) * MTE(U)] / E[w(U)]`` as a posterior
        quantity by reusing the fit's posterior draws on ``b_mte``
        and the stored ``u_grid`` / ``poly_u``. The numerator and
        denominator are evaluated on the same grid so integration
        weights cancel.

        Parameters
        ----------
        weight_fn : callable
            Vectorised function ``u -> weights``, where ``u`` is a
            numpy array on ``self.u_grid``. Values outside ``[0, 1]``
            are still valid (the grid dictates the support).
        label : str
            Identifier propagated into the returned summary dict.
        rope : (float, float), optional
            Region of practical equivalence for the policy effect.

        Returns
        -------
        dict
            Keys: ``label, estimate, std_error, hdi_low, hdi_high,
            prob_positive``, plus ``prob_rope`` if ``rope`` given.
        """
        if self.trace is None or self.u_grid is None:
            raise MethodIncompatibility(
                "policy_effect requires the fit's trace and u_grid.",
                recovery_hint="Call policy_effect() on a fitted "
                "BayesianMTEResult rather than a serialized summary.",
                diagnostics={
                    "has_trace": self.trace is not None,
                    "has_u_grid": self.u_grid is not None,
                },
                alternative_functions=["sp.bayes_mte"],
            )
        if not callable(weight_fn):
            raise MethodIncompatibility(
                "weight_fn must be callable.",
                recovery_hint="Pass a function that accepts u_grid and returns "
                "one weight per grid point.",
                diagnostics={"type": type(weight_fn).__name__},
                alternative_functions=["sp.bayes_mte"],
            )
        b_mte_post = self.trace.posterior["b_mte"].values
        poly_u = b_mte_post.shape[-1] - 1
        flat = b_mte_post.reshape(-1, poly_u + 1)
        u = np.asarray(self.u_grid, dtype=float)
        if u.ndim != 1 or len(u) == 0:
            raise DataInsufficient(
                "u_grid must be a non-empty one-dimensional array.",
                recovery_hint="Refit sp.bayes_mte() with a valid u_grid.",
                diagnostics={"shape": u.shape},
                alternative_functions=["sp.bayes_mte"],
            )
        if not np.all(np.isfinite(u)):
            raise NumericalInstability(
                "u_grid contains NaN or infinite values.",
                recovery_hint="Refit sp.bayes_mte() with a finite propensity grid.",
                diagnostics={"nonfinite_count": int((~np.isfinite(u)).sum())},
                alternative_functions=["sp.bayes_mte"],
            )

        # v0.9.12: respect the selection scale. Under
        # ``selection='normal'`` the polynomial is in ``v = Φ^{-1}(u)``
        # so the abscissa-powers must be built on v, not u. The
        # weight_fn is still passed u (the natural scale for users)
        # but we transform internally before dotting with b_mte.
        if self.selection == "normal":
            from scipy.stats import norm as _norm_dist

            abscissa = _norm_dist.ppf(np.clip(u, PROBIT_CLIP, 1 - PROBIT_CLIP))
        else:
            abscissa = u
        u_pow = np.column_stack([abscissa**k for k in range(poly_u + 1)])
        mte_samples = flat @ u_pow.T  # (S, n_grid)
        try:
            weights = np.asarray(weight_fn(u), dtype=float)
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "weight_fn must return numeric weights.",
                recovery_hint="Return a numeric array with the same shape as "
                "the provided u_grid.",
                diagnostics={"u_shape": u.shape},
                alternative_functions=["sp.bayes_mte"],
            ) from exc
        if weights.shape != u.shape:
            raise MethodIncompatibility(
                f"weight_fn must return an array of shape {u.shape}; "
                f"got {weights.shape}.",
                recovery_hint="Return exactly one weight per u_grid point.",
                diagnostics={"expected_shape": u.shape, "actual_shape": weights.shape},
                alternative_functions=["sp.bayes_mte"],
            )
        if not np.all(np.isfinite(weights)):
            raise NumericalInstability(
                "weight_fn produced NaN or infinite weights.",
                recovery_hint="Clip or redefine the policy weights so they are "
                "finite on the fitted u_grid.",
                diagnostics={
                    "nonfinite_count": int((~np.isfinite(weights)).sum()),
                },
                alternative_functions=["sp.bayes_mte"],
            )
        if not np.any(weights):
            raise MethodIncompatibility(
                "weight_fn produced all-zero weights on the grid.",
                recovery_hint="Use weights with positive support on the fitted "
                "u_grid.",
                diagnostics={"n_grid": int(len(u))},
                alternative_functions=["sp.bayes_mte"],
            )

        # Trapezoidal integration on ``u_grid`` for both the numerator
        # ``int w(u) MTE(u) du`` and the denominator ``int w(u) du``.
        # Using trapezoid (rather than a simple sum) makes
        # ``policy_effect(policy_weight_ate())`` numerically identical
        # to the internally-stored ``.ate`` field (both use the same
        # integrator). The sum-based normalisation is a valid
        # approximation on a uniform grid but diverges from .ate by
        # the endpoint-half-weight correction; matching .ate is
        # more important for agent-native parity.
        denom = float(np.trapezoid(weights, x=u))
        if denom == 0.0:
            raise MethodIncompatibility(
                "weight_fn produced an integrated weight of 0 on the grid.",
                recovery_hint="Use weights with non-zero integrated mass.",
                diagnostics={"denominator": denom},
                alternative_functions=["sp.bayes_mte"],
            )
        numer_samples = np.trapezoid(mte_samples * weights, x=u, axis=1)
        policy_samples = numer_samples / denom

        hdi = _az_hdi_compat(policy_samples, hdi_prob=self.hdi_prob)
        summary = {
            "label": label,
            "estimate": float(np.mean(policy_samples)),
            "std_error": float(np.std(policy_samples, ddof=1)),
            "hdi_low": float(hdi[0]),
            "hdi_high": float(hdi[1]),
            "prob_positive": float(np.mean(policy_samples > 0)),
        }
        rope = _validate_rope(rope)
        if rope is not None:
            lo, hi = rope
            summary["prob_rope"] = float(
                np.mean((policy_samples > lo) & (policy_samples < hi))
            )
        return summary

    def plot_mte(
        self,
        ax: Any = None,
        figsize: Tuple[float, float] = (8, 5),
    ) -> Tuple[Any, Any]:
        """Plot the MTE curve with HDI ribbon. Requires matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("plot_mte requires matplotlib; `pip install matplotlib`.")
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        curve = self.mte_curve
        ax.plot(
            curve["u"],
            curve["posterior_mean"],
            color="#2C3E50",
            linewidth=2,
            label="posterior mean",
        )
        ax.fill_between(
            curve["u"],
            curve["hdi_low"],
            curve["hdi_high"],
            color="#2C3E50",
            alpha=0.2,
            label=f"{int(self.hdi_prob * 100)}% HDI",
        )
        ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_xlabel("Propensity to be treated ($U_D$)", fontsize=11)
        ax.set_ylabel("MTE($u$)", fontsize=11)
        ax.set_title("Marginal Treatment Effect", fontsize=13)
        ax.legend(fontsize=9, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return fig, ax


def _summarise_posterior(
    trace: Any,
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
    hdi_arr = _az_hdi_compat(posterior, hdi_prob=hdi_prob)
    hdi_lower = float(hdi_arr[0])
    hdi_upper = float(hdi_arr[1])
    prob_positive = float(np.mean(posterior > 0))

    # Convergence diagnostics
    try:
        rhat = float(az.rhat(trace, var_names=[var_name])[var_name].values)
    except Exception:
        rhat = float("nan")
    try:
        ess = float(az.ess(trace, var_names=[var_name])[var_name].values)
    except Exception:
        ess = float("nan")

    out = {
        "posterior_mean": mean,
        "posterior_median": median,
        "posterior_sd": sd,
        "hdi_lower": hdi_lower,
        "hdi_upper": hdi_upper,
        "prob_positive": prob_positive,
        "rhat": rhat,
        "ess": ess,
    }
    if rope is not None:
        lo, hi = rope
        out["prob_rope"] = float(np.mean((posterior > lo) & (posterior < hi)))
    return out
