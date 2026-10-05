"""
Comparing Bayesian models: Bayes factors, posterior model probabilities
and the Savage-Dickey density ratio.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import linalg, special, stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility, StatsPAIWarning

_LOG_2PI = float(np.log(2.0 * np.pi))


def _evidence_label(two_log_bf: float) -> str:
    """Kass and Raftery (1995) verbal scale for ``2 log BF``."""
    v = abs(two_log_bf)
    if v < 2:
        return "not worth more than a bare mention"
    if v < 6:
        return "positive"
    if v < 10:
        return "strong"
    return "very strong"


@dataclass
class BayesFactorResult(ResultProtocolMixin):
    """Result of :func:`bayes_factor`.

    Attributes
    ----------
    table : pd.DataFrame
        One row per model: ``log_marglik``, ``prior_prob``, ``post_prob``
        and ``log_bf`` (log Bayes factor against the first model, so
        positive values favour the row's model).
    log_bf : float
        Log Bayes factor of the first model against the second.
    bf : float
        The Bayes factor itself (``inf`` when it overflows).
    evidence : str
        Kass and Raftery's label for ``2 log BF`` and the model it
        favours.

    Examples
    --------
    >>> import statspai as sp
    >>> out = sp.bayes_factor(-100.0, -103.0)
    >>> round(out.log_bf, 1)
    3.0
    >>> out.evidence
    'strong evidence for model 1'
    """

    table: pd.DataFrame
    log_bf: float
    bf: float
    evidence: str
    methods: Dict[str, str] = field(default_factory=dict)

    _citation_keys = ("kass1995bayes",)

    def summary(self) -> str:
        lines = [
            "Bayes factor",
            self.table.to_string(float_format=lambda v: f"{v:.5g}"),
            "",
            f"log BF (first vs second) = {self.log_bf:.4f}    "
            f"2 log BF = {2 * self.log_bf:.4f}",
            f"{self.evidence}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def bayes_factor(
    *models: Any,
    names: Optional[Sequence[str]] = None,
    prior_probs: Optional[Sequence[float]] = None,
    method: Optional[str] = None,
) -> BayesFactorResult:
    """Bayes factors and posterior model probabilities.

    Parameters
    ----------
    *models
        Two or more fitted models that have a
        ``log_marginal_likelihood()`` method (the results of
        :func:`statspai.bayes_regress`), or their log marginal
        likelihoods as numbers.
    names : list of str, optional
        Labels for the models.
    prior_probs : list of float, optional
        Prior model probabilities; equal by default.
    method : str, optional
        Passed to each model's ``log_marginal_likelihood``.

    Returns
    -------
    BayesFactorResult

    Notes
    -----
    Bayes factors require proper priors and are sensitive to them: making
    the prior of an extra coefficient more diffuse favours the smaller
    model without bound (Lindley's paradox). They compare models for the
    same outcome on the same observations; the function refuses fitted
    models with different numbers of observations.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=100), "z": rng.normal(size=100)})
    >>> df["y"] = 1 + df["x"] + rng.normal(size=100)
    >>> m1 = sp.bayes_regress("y ~ x", df, model="conjugate", seed=1)
    >>> m2 = sp.bayes_regress("y ~ x + z", df, model="conjugate", seed=1)
    >>> bool(sp.bayes_factor(m1, m2).log_bf > 0)
    True

    References
    ----------
    kass1995bayes
    """
    if len(models) < 2:
        raise MethodIncompatibility("bayes_factor needs at least two models.")
    lml: List[float] = []
    used: Dict[str, str] = {}
    n_obs = []
    labels = (
        list(names)
        if names is not None
        else [f"model {i + 1}" for i in range(len(models))]
    )
    if len(labels) != len(models):
        raise MethodIncompatibility("names must have one label per model.")
    for lab, m in zip(labels, models):
        if isinstance(m, (int, float, np.floating)):
            lml.append(float(m))
            used[lab] = "supplied"
        elif hasattr(m, "marginal_likelihood_details"):
            d = m.marginal_likelihood_details(method)
            lml.append(float(d["log_marginal_likelihood"]))
            used[lab] = str(d["method"])
            n_obs.append(int(m.n_obs))
        elif hasattr(m, "log_marginal_likelihood"):
            val = m.log_marginal_likelihood
            lml.append(float(val() if callable(val) else val))
            used[lab] = "model"
        else:
            raise MethodIncompatibility(
                f"{lab} is neither a number nor a fitted model with a "
                "log_marginal_likelihood."
            )
    if len(set(n_obs)) > 1:
        raise MethodIncompatibility(
            "The models were fitted on different numbers of observations "
            f"({sorted(set(n_obs))}); marginal likelihoods are comparable "
            "only on the same data. Drop rows with missing values in any "
            "model's variables before fitting."
        )
    arr = np.asarray(lml)
    if not np.isfinite(arr).all():
        raise MethodIncompatibility("A log marginal likelihood is not finite.")
    if prior_probs is None:
        pp: np.ndarray = np.full(arr.size, 1.0 / arr.size)
    else:
        pp = np.asarray(prior_probs, dtype=float)
        if pp.shape != arr.shape or np.any(pp <= 0):
            raise MethodIncompatibility("prior_probs must be positive, one per model.")
        pp = pp / pp.sum()
    logpost = arr + np.log(pp)
    post = np.exp(logpost - special.logsumexp(logpost))
    table = pd.DataFrame(
        {
            "log_marglik": arr,
            "prior_prob": pp,
            "post_prob": post,
            "log_bf": arr - arr[0],
        },
        index=labels,
    )
    log_bf = float(arr[0] - arr[1])
    with np.errstate(over="ignore"):
        bf = float(np.exp(log_bf))
    favoured = labels[0] if log_bf >= 0 else labels[1]
    label = _evidence_label(2.0 * log_bf)
    evidence = (
        f"{label} evidence for {favoured}"
        if label != "not worth more than a bare mention"
        else f"evidence for {favoured} {label}"
    )
    return BayesFactorResult(
        table=table, log_bf=log_bf, bf=bf, evidence=evidence, methods=used
    )


def savage_dickey(result: Any, param: str, value: float = 0.0) -> Dict[str, Any]:
    """Savage-Dickey density ratio for a point restriction.

    The Bayes factor of the restriction ``param = value`` against the
    fitted (unrestricted) model equals the posterior density at
    ``value`` divided by the prior density there (Dickey 1971), provided
    the prior of the other parameters under the restriction is their
    conditional prior given ``param = value``. With the independent
    priors of :func:`statspai.bayes_regress` that holds when the prior
    covariance is diagonal. Under ``model='conjugate'`` the coefficients'
    prior is scaled by ``sigma2``, so conditioning on ``param = value``
    also updates ``sigma2``: the ratio is then the Bayes factor against
    the restricted model whose prior is
    ``sigma2 ~ InvGamma((alpha0 + 1) / 2, (delta0 + (value - b0)^2 / B0) / 2)``,
    not ``InvGamma(alpha0 / 2, delta0 / 2)``.

    Parameters
    ----------
    result : BayesRegressResult
    param : str
        A coefficient of the model.
    value : float, default 0

    Returns
    -------
    dict
        ``bf01`` (restriction against unrestricted), ``log_bf01``,
        ``posterior_density``, ``prior_density``, ``estimator`` (how the
        posterior density was obtained) and ``evidence``.

    Notes
    -----
    The posterior density is exact for the conjugate model
    (Student-t), a Rao-Blackwell average of the full conditionals for the
    normal and probit models, and a kernel estimate from the draws
    otherwise. A kernel estimate is unreliable far in the tail; the
    function then reports the density of the normal approximation and
    says so.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=200), "z": rng.normal(size=200)})
    >>> df["y"] = 1 + df["x"] + rng.normal(size=200)
    >>> fit = sp.bayes_regress("y ~ x + z", df, model="conjugate", seed=1)
    >>> bool(sp.savage_dickey(fit, "z")["bf01"] > 1)
    True

    References
    ----------
    dickey1971weighted, verdinelli1995computing
    """
    mdl = getattr(result, "_model", None)
    if mdl is None or not hasattr(result, "draws"):
        raise MethodIncompatibility("savage_dickey needs a result of sp.bayes_regress.")
    xnames = list(getattr(mdl, "xnames", []))
    if result.model == "oprobit":
        raise MethodIncompatibility(
            "savage_dickey is not available for model='oprobit', whose "
            "reported parameters are a transformation of the ones the "
            "prior is stated for. Compare marginal likelihoods with "
            "sp.bayes_factor instead."
        )
    if param not in xnames:
        raise MethodIncompatibility(
            f"{param!r} is not a coefficient of the model; choose from " f"{xnames}."
        )
    j = xnames.index(param)
    b0, B0 = mdl.b0, mdl.B0
    offdiag = B0[j].copy()
    offdiag[j] = 0.0
    if np.any(np.abs(offdiag) > 1e-12 * B0[j, j]):
        warnings.warn(
            f"The prior of {param!r} is correlated with other coefficients; "
            "the Savage-Dickey ratio then tests the restriction with the "
            "conditional prior of the others, which is not the prior of a "
            "model fitted without the term.",
            StatsPAIWarning,
            stacklevel=2,
        )
    draws = result.draws[param].to_numpy()
    if result.model == "conjugate":
        scale_prior = np.sqrt(mdl.d0 / mdl.a0 * B0[j, j])
        prior_den = float(
            stats.t.pdf((value - b0[j]) / scale_prior, df=mdl.a0) / scale_prior
        )
        scale_post = np.sqrt(mdl.dn / mdl.an * mdl.Bn[j, j])
        post_den = float(
            stats.t.pdf((value - mdl.bn[j]) / scale_post, df=mdl.an) / scale_post
        )
        estimator = "exact"
    else:
        prior_den = float(stats.norm.pdf(value, b0[j], np.sqrt(B0[j, j])))
        if result.model == "normal":
            s2 = result.draws["sigma2"].to_numpy()
            logs = np.empty(s2.size)
            for g, v in enumerate(s2):
                mean, prec = mdl.beta_conditional(float(v))
                var_j = linalg.inv(prec)[j, j]
                logs[g] = (
                    -0.5 * (_LOG_2PI + np.log(var_j))
                    - 0.5 * (value - mean[j]) ** 2 / var_j
                )
            post_den = float(np.exp(special.logsumexp(logs) - np.log(logs.size)))
            estimator = "rao-blackwell"
        elif result.model == "probit" and "cond_mean" in getattr(result, "_extras", {}):
            cm = result._extras["cond_mean"][:, j]
            var_j = linalg.cho_solve((mdl.Bn_chol, True), np.eye(mdl.k))[j, j]
            logs = -0.5 * (_LOG_2PI + np.log(var_j)) - 0.5 * (value - cm) ** 2 / var_j
            post_den = float(np.exp(special.logsumexp(logs) - np.log(logs.size)))
            estimator = "rao-blackwell"
        else:
            tail = min((draws <= value).mean(), (draws >= value).mean())
            if tail * draws.size >= 50:
                post_den = float(stats.gaussian_kde(draws)(value)[0])
                estimator = "kernel"
            else:
                post_den = float(stats.norm.pdf(value, draws.mean(), draws.std(ddof=1)))
                estimator = "normal approximation (value is in the tail of the draws)"
    with np.errstate(divide="ignore"):
        log_bf01 = float(np.log(post_den) - np.log(prior_den))
    label = _evidence_label(2.0 * log_bf01)
    side = f"{param} = {value:g}" if log_bf01 >= 0 else f"{param} != {value:g}"
    evidence = (
        f"{label} evidence for {side}"
        if label != "not worth more than a bare mention"
        else f"evidence for {side} {label}"
    )
    return {
        "param": param,
        "value": float(value),
        "bf01": float(np.exp(log_bf01)) if np.isfinite(log_bf01) else 0.0,
        "log_bf01": log_bf01,
        "posterior_density": post_den,
        "prior_density": prior_den,
        "estimator": estimator,
        "evidence": evidence,
    }
