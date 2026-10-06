"""
Bayesian regression by MCMC: ``sp.bayes_regress``.

One entry point for the single-equation models of an introductory
Bayesian econometrics course: the Gaussian linear model (conjugate, or
with independent priors), Student-t errors, logit, probit, ordered
probit, Poisson, negative binomial, tobit and quantile regression. The
samplers are plain NumPy (Gibbs, data augmentation, random-walk
Metropolis), so nothing beyond the core dependencies is needed. For the
causal designs (DiD, RD, IV, ...) with PyMC see :mod:`statspai.bayes`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import linalg, special, stats

from .._result_serialize import ResultProtocolMixin
from ..core.utils import create_design_matrices
from ..exceptions import (
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
    StatsPAIWarning,
)
from . import _models as M
from . import _workflow as W
from ._core import check_mcmc_args, design_for, normal_prior, spawn_rngs
from .diagnostics import (
    _spectrum0_ar,
    gelman_rubin,
    geweke_diag,
    heidel_diag,
    hpd_interval,
    mcmc_summary,
    raftery_diag,
)

_MODEL_ALIASES = {
    "gaussian": "normal",
    "linear": "normal",
    "regress": "normal",
    "ols": "normal",
    "student": "t",
    "student-t": "t",
    "ordered_probit": "oprobit",
    "ordered-probit": "oprobit",
    "nbreg": "negbin",
    "negative_binomial": "negbin",
    "qreg": "quantile",
    "binomial": "logit",
}

_CITATIONS = {
    "normal": ("gelfand1990sampling",),
    "conjugate": ("ramirezhassan2026introduction",),
    "t": ("geweke1993bayesian",),
    "probit": ("albert1993bayesian",),
    "oprobit": ("albert1993bayesian", "cowles1996accelerating"),
    "tobit": ("chib1992bayes",),
    "quantile": ("kozumi2011gibbs",),
    "logit": ("metropolis1953equation", "hastings1970monte"),
    "poisson": ("metropolis1953equation", "hastings1970monte"),
    "negbin": ("metropolis1953equation", "hastings1970monte"),
    "bayes_bootstrap": ("rubin1981bayesian",),
    "mlogit": ("metropolis1953equation", "hastings1970monte"),
    "sur": ("zellner1962efficient", "rossi2005bayesian"),
    "lasso": ("park2008bayesian",),
    "ssvs": ("george1993variable",),
    "iv": ("rossi2005bayesian",),
    "stochvol": ("kastner2014ancillarity",),
    "mvprobit": ("rossi2005bayesian", "albert1993bayesian"),
    "mnprobit": ("mcculloch1994exact", "rossi2005bayesian"),
    "mixture": ("neal2000markov", "escobar1995bayesian"),
    "abc": ("beaumont2002approximate", "wood2010statistical"),
    "arima": ("metropolis1953equation", "hastings1970monte"),
}


class _ConjugateNormal:
    """Normal / inverse-gamma conjugate regression, in closed form.

    ``beta | sigma2 ~ N(b0, sigma2 B0)``, ``sigma2 ~ IG(a0/2, d0/2)``.
    """

    name = "conjugate"
    sampler = "exact (independent draws from the closed-form posterior)"

    def __init__(
        self,
        y: Any,
        X: Any,
        xnames: Any,
        prior_mean: Any,
        prior_var: Any,
        a0: Any,
        d0: Any,
    ) -> None:
        self.y = np.asarray(y, dtype=float)
        self.X = np.asarray(X, dtype=float)
        self.n, self.k = self.X.shape
        self.xnames = [str(c) for c in xnames]
        self.b0, self.B0, self.B0inv = normal_prior(
            self.k, prior_mean, prior_var, self.xnames
        )
        if a0 <= 0 or d0 <= 0:
            raise MethodIncompatibility(
                "sigma2_prior must be two positive numbers (alpha0, delta0)."
            )
        self.a0, self.d0 = float(a0), float(d0)
        prec = self.B0inv + self.X.T @ self.X
        self.prec_chol = linalg.cholesky(prec, lower=True)
        self.Bn = linalg.cho_solve((self.prec_chol, True), np.eye(self.k))
        self.Bn = 0.5 * (self.Bn + self.Bn.T)
        rhs = self.B0inv @ self.b0 + self.X.T @ self.y
        self.bn = self.Bn @ rhs
        self.an = self.a0 + self.n
        self.dn = float(
            self.d0 + self.y @ self.y + self.b0 @ self.B0inv @ self.b0 - self.bn @ rhs
        )
        self.names = self.xnames + ["sigma2"]

    def log_marginal_likelihood(self) -> float:
        logdet_bn = -2.0 * np.log(np.diag(self.prec_chol)).sum()
        logdet_b0 = np.linalg.slogdet(self.B0)[1]
        return float(
            -0.5 * self.n * np.log(np.pi)
            + 0.5 * self.a0 * np.log(self.d0)
            - 0.5 * self.an * np.log(self.dn)
            + 0.5 * (logdet_bn - logdet_b0)
            + special.gammaln(self.an / 2.0)
            - special.gammaln(self.a0 / 2.0)
        )

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        s2 = (self.dn / 2.0) / rng.gamma(self.an / 2.0, size=n)
        chol = linalg.cholesky(self.Bn, lower=True)
        z = rng.standard_normal((n, self.k)) @ chol.T
        beta = self.bn[None, :] + np.sqrt(s2)[:, None] * z
        return np.column_stack([beta, s2])

    def exact_table(self, level: float) -> pd.DataFrame:
        """Posterior moments and quantiles from the closed form."""
        scale = np.sqrt(self.dn / self.an * np.diag(self.Bn))
        lo, hi = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
        tdist = stats.t(df=self.an)
        mean = np.append(self.bn, self.dn / (self.an - 2.0))
        sd_beta = scale * np.sqrt(self.an / (self.an - 2.0))
        sd_s2 = (self.dn / 2.0) / ((self.an / 2.0 - 1.0) * np.sqrt(self.an / 2.0 - 2.0))
        ig = stats.invgamma(self.an / 2.0, scale=self.dn / 2.0)
        return pd.DataFrame(
            {
                "mean": mean,
                "sd": np.append(sd_beta, sd_s2),
                "lower": np.append(self.bn + scale * tdist.ppf(lo), ig.ppf(lo)),
                "median": np.append(self.bn, ig.ppf(0.5)),
                "upper": np.append(self.bn + scale * tdist.ppf(hi), ig.ppf(hi)),
                "prob_positive": np.append(tdist.cdf(self.bn / scale), 1.0),
            },
            index=self.names,
        )

    def linear_predictor(self, draws: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.asarray(draws[:, : self.k] @ X.T)

    @staticmethod
    def expected_value(eta: np.ndarray) -> np.ndarray:
        return eta


@dataclass
class _OffsetSpec:
    """The known term of the linear index and how to rebuild it."""

    values: np.ndarray
    column: Optional[str]
    log: bool
    label: str

    @classmethod
    def build(
        cls,
        offset: Any,
        exposure: Optional[str],
        data: pd.DataFrame,
        frame: pd.DataFrame,
        model_key: str,
    ) -> Optional["_OffsetSpec"]:
        if offset is None and exposure is None:
            return None
        if offset is not None and exposure is not None:
            raise MethodIncompatibility("Pass offset or exposure, not both.")
        if model_key not in ("logit", "poisson", "negbin"):
            raise MethodIncompatibility(
                "offset / exposure are for the logit, poisson and negbin "
                f"models; got model='{model_key}'."
            )
        if exposure is not None and model_key == "logit":
            raise MethodIncompatibility("exposure is for count models.")
        name = exposure if exposure is not None else offset
        take_log = exposure is not None
        if isinstance(name, str):
            if name not in data.columns:
                raise MethodIncompatibility(f"Column {name!r} is not in data.")
            raw = frame[name].to_numpy(dtype=float)
            column: Optional[str] = name
            label = f"log({name})" if take_log else name
        else:
            arr = np.asarray(name, dtype=float).reshape(-1)
            if arr.size != len(data):
                raise MethodIncompatibility(
                    f"offset has {arr.size} entries for {len(data)} rows of data."
                )
            raw = pd.Series(arr, index=data.index).loc[frame.index].to_numpy()
            column, label = None, "array"
        if take_log:
            if np.any(raw <= 0):
                raise MethodIncompatibility("exposure must be positive.")
            raw = np.log(raw)
        if not np.all(np.isfinite(raw)):
            raise MethodIncompatibility(
                "The offset has missing or infinite values on the estimation " "sample."
            )
        return cls(values=raw, column=column, log=take_log, label=label)

    def for_data(self, data: pd.DataFrame) -> np.ndarray:
        if self.column is None:
            raise MethodIncompatibility(
                "The offset was passed as an array, so it cannot be rebuilt "
                "for new data. Fit with offset='<column>'."
            )
        if self.column not in data.columns:
            raise MethodIncompatibility(
                f"The new data lack the offset column {self.column!r}."
            )
        raw = data[self.column].to_numpy(dtype=float)
        return np.asarray(np.log(raw) if self.log else raw)


class _PredictiveMethods:
    """Predictive distribution and pointwise likelihood of a fitted model."""

    _model: Any
    _offset: Any
    _frame: Any
    _call: Dict[str, Any]
    _design_info: Any
    draws: pd.DataFrame
    formula: str
    model: str
    #: provided by the result class
    _design: Any

    def _offset_for(self, data: Optional[pd.DataFrame]) -> Any:
        if self._offset is None:
            return 0.0
        if data is None:
            return self._offset.values
        return self._offset.for_data(data)

    def _linear_index(
        self, draws: np.ndarray, X: np.ndarray, data: Optional[pd.DataFrame]
    ) -> np.ndarray:
        """Linear index by draw and row, with the offset when there is one.

        Goes through the model's own ``linear_predictor`` so that the
        result classes built on a shim (mixed models, systems) keep
        working.
        """
        eta = self._model.linear_predictor(draws, X)
        if self._offset is None:
            return np.asarray(eta)
        return np.asarray(eta + self._offset_for(data))

    def _predictive_model(self, what: str) -> Any:
        mdl = self._model
        if mdl is None or getattr(mdl, "name", None) not in W.PREDICTIVE_MODELS:
            raise MethodIncompatibility(
                f"{what} is available for the fits of sp.bayes_regress and "
                "sp.bayes_shrink, not for this model."
            )
        return mdl

    def _draw_rows(self, index: Optional[Any]) -> np.ndarray:
        d = self.draws.to_numpy()
        return np.asarray(d if index is None else d[np.asarray(index)])

    def _outcome_and_rows(self, data: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Outcome of new data on the model's scale, and the rows used."""
        if self.model in ("oprobit", "mlogit"):
            lhs = self.formula.split("~", 1)[0].strip()
            if lhs not in data.columns:
                raise MethodIncompatibility(f"The new data lack the outcome {lhs!r}.")
            rows = data.loc[data[lhs].notna()]
            levels = list(self._model.levels)
            codes = pd.Categorical(rows[lhs], categories=levels).codes
            if np.any(codes < 0):
                raise MethodIncompatibility(
                    "The new data have outcome values the model was not "
                    f"fitted to; known levels: {levels}."
                )
            return np.asarray(codes, dtype=float), rows
        y_df, _ = create_design_matrices(self.formula, data)
        rows = data.loc[y_df.index]
        return np.asarray(y_df, dtype=float).reshape(-1), rows

    def posterior_linpred(
        self, data: Optional[pd.DataFrame] = None, index: Optional[Any] = None
    ) -> np.ndarray:
        """Draws of the linear index: one row per draw, one column per row
        of ``data`` (the estimation sample when omitted). ``index``
        selects draws."""
        X = self._design(data)
        return np.asarray(self._linear_index(self._draw_rows(index), X, data))

    def posterior_epred(
        self, data: Optional[pd.DataFrame] = None, index: Optional[Any] = None
    ) -> np.ndarray:
        """Draws of the expected outcome ``E[y | x]``: the probability for
        binary models, the rate for count models. Uncertainty about the
        coefficients only; see :meth:`posterior_predict` for new
        outcomes."""
        return np.asarray(
            self._model.expected_value(self.posterior_linpred(data, index))
        )

    def posterior_predict(
        self,
        data: Optional[pd.DataFrame] = None,
        draws: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draws of new outcomes from the posterior predictive distribution.

        One simulated outcome per posterior draw and row of ``data``:
        coefficient uncertainty plus the observation noise of the model.
        Ordered and multinomial outcomes come back as positions in the
        list of levels.

        Parameters
        ----------
        data : DataFrame, optional
            New data; the estimation sample when omitted.
        draws : int, optional
            Use this many posterior draws, evenly spaced; all by default.
        seed : int, optional
        """
        total = len(self.draws)
        index = None
        if draws is not None:
            if not 1 <= int(draws) <= total:
                raise MethodIncompatibility(
                    f"draws must be between 1 and {total}; got {draws}."
                )
            index = np.unique(np.linspace(0, total - 1, int(draws)).round().astype(int))
        mdl = self._predictive_model("posterior_predict()")
        X = self._design(data)
        rng = np.random.default_rng(seed)
        return W.predictive_draws(
            mdl, self._draw_rows(index), X, rng, self._offset_for(data)
        )

    def log_lik(
        self, data: Optional[pd.DataFrame] = None, index: Optional[Any] = None
    ) -> np.ndarray:
        """Pointwise log-likelihood: one row per draw, one column per
        observation of ``data`` (the estimation sample when omitted).

        The input of :func:`statspai.loo` and :func:`statspai.waic`. For
        new data the outcome column must be present; rows with missing
        values are dropped.
        """
        mdl = self._predictive_model("log_lik()")
        d = self._draw_rows(index)
        if data is None:
            y = mdl.yi if self.model in ("oprobit", "mlogit") else mdl.y
            return W.pointwise_log_lik(mdl, d, y, mdl.X, self._offset_for(None))
        y, rows = self._outcome_and_rows(data)
        X = self._design(rows)
        return W.pointwise_log_lik(mdl, d, y, X, self._offset_for(rows))

    def _refit(self, data: pd.DataFrame) -> Any:
        """The same model fitted to other data (cross-validation)."""
        call = dict(self._call)
        if call.get("offset") is not None and not isinstance(call["offset"], str):
            raise MethodIncompatibility(
                "Refitting needs the offset as a column name, not an array."
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StatsPAIWarning)
            return bayes_regress(self.formula, data, **call)


@dataclass
class BayesRegressResult(_PredictiveMethods, ResultProtocolMixin):
    """Posterior of a model fitted by :func:`bayes_regress`.

    Attributes
    ----------
    model : str
        The likelihood (``'normal'``, ``'logit'``, ...).
    params : pd.Series
        Posterior means.
    std_errors : pd.Series
        Posterior standard deviations.
    table : pd.DataFrame
        One row per parameter: posterior mean, standard deviation, Monte
        Carlo standard error of the mean (``mcse``), effective sample
        size, the equal-tailed credible interval, the median and the
        posterior probability of a positive value.
    draws : pd.DataFrame
        The retained draws, chains stacked, one column per parameter.
    chain : np.ndarray
        Chain label of every row of ``draws``.
    acceptance_rate : float or None
        Metropolis acceptance rate, when the sampler has such a step.
    prior : dict
        The prior actually used.
    n_obs : int
    level : float
        Mass of the credible interval in ``table``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=200)})
    >>> df["y"] = 1 + 0.5 * df["x"] + rng.normal(size=200)
    >>> fit = sp.bayes_regress("y ~ x", df, draws=1000, burnin=200, seed=1)
    >>> isinstance(fit, sp.BayesRegressResult)
    True
    >>> list(fit.params.index)
    ['Intercept', 'x', 'sigma2']
    >>> fit.conf_int().shape
    (3, 2)
    """

    model: str
    formula: str
    params: pd.Series
    std_errors: pd.Series
    table: pd.DataFrame
    draws: pd.DataFrame
    chain: np.ndarray
    n_obs: int
    n_draws: int
    burnin: int
    thin: int
    chains: int
    sampler: str
    acceptance_rate: Optional[float]
    prior: Dict[str, Any]
    level: float = 0.95
    model_info: Dict[str, Any] = field(default_factory=dict)
    diagnostics_info: Dict[str, Any] = field(default_factory=dict)
    _model: Any = field(default=None, repr=False)
    _extras: Dict[str, Any] = field(default_factory=dict, repr=False)
    _design_info: Any = field(default=None, repr=False)
    _lml_cache: Dict[str, Dict[str, float]] = field(default_factory=dict, repr=False)
    _frame: Any = field(default=None, repr=False)
    _call: Dict[str, Any] = field(default_factory=dict, repr=False)
    _offset: Any = field(default=None, repr=False)

    #: every paper a model of this family rests on; ``cite()`` returns the
    #: ones of the fitted model
    _citation_keys = tuple(sorted({k for v in _CITATIONS.values() for k in v}))

    def cite(self, format: str = "keys") -> Any:  # noqa: A002
        """The verified ``paper.bib`` keys behind the fitted model."""
        keys = list(_CITATIONS.get(self.model, self._citation_keys))
        if format == "json":
            return {
                "citation_keys": keys,
                "source": "paper.bib",
                "resolve_with": "sp.bibtex(keys=[...])",
            }
        if format != "keys":
            raise MethodIncompatibility(
                f"format must be 'keys' or 'json'; got {format!r}"
            )
        return "\n".join(keys)

    # -- basic accessors ---------------------------------------------------
    @property
    def coef(self) -> pd.Series:
        return self.params

    def conf_int(
        self, level: Optional[float] = None, kind: str = "equal"
    ) -> pd.DataFrame:
        """Credible intervals.

        Parameters
        ----------
        level : float, optional
            Posterior mass; defaults to the level the model was fitted
            with.
        kind : {'equal', 'hpd'}
            Equal-tailed (quantiles) or highest posterior density.
        """
        level = self.level if level is None else level
        if kind == "hpd":
            return hpd_interval(self.draws, prob=level)
        if kind != "equal":
            raise MethodIncompatibility("kind must be 'equal' or 'hpd'.")
        if self.model == "conjugate" and level == self.level:
            return self.table[["lower", "upper"]].copy()
        if self.model == "conjugate":
            return self._model.exact_table(level)[["lower", "upper"]]
        lo = (1.0 - level) / 2.0
        q = self.draws.quantile([lo, 1.0 - lo]).T
        q.columns = ["lower", "upper"]
        return q

    def prob(self, expr: str) -> float:
        """Posterior probability of a statement about the parameters.

        ``expr`` is evaluated on the draws with the parameter names as
        variables, e.g. ``fit.prob("x > 0")`` or
        ``fit.prob("x1 > x2")``. Names that are not valid Python
        identifiers are available through backticks, as in
        ``DataFrame.eval``.
        """
        try:
            val = self.draws.eval(expr)
        except (SyntaxError, NameError, KeyError, TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                f"Could not evaluate {expr!r} on the draws: {exc}. Available "
                f"parameters: {list(self.draws.columns)}."
            ) from exc
        arr = np.asarray(val)
        if arr.dtype != bool:
            raise MethodIncompatibility(
                f"{expr!r} is not a true / false statement about the "
                "parameters (e.g. 'x > 0')."
            )
        return float(arr.mean())

    # -- summaries ---------------------------------------------------------
    def summary(self) -> str:
        pct = f"{100 * self.level:g}%"
        lines = [
            f"Bayesian {self.model} regression    {self.formula}",
            f"Observations: {self.n_obs}    Sampler: {self.sampler}",
        ]
        if self.model in ("conjugate", "bayes_bootstrap"):
            lines.append(f"{self.n_draws} independent draws.")
        else:
            lines.append(
                f"Chains: {self.chains}    Draws kept: {self.n_draws}    "
                f"Burn-in: {self.burnin}    Thinning: {self.thin}"
            )
        if self.acceptance_rate is not None:
            lines.append(f"Acceptance rate: {self.acceptance_rate:.3f}")
        lines.append("")
        show = self.table.rename(
            columns={
                "lower": f"[{pct}",
                "upper": "interval]",
                "prob_positive": "P(>0)",
            }
        )
        lines.append(show.to_string(float_format=lambda v: f"{v:.5g}"))
        notes = self.diagnostics_info.get("warnings") or []
        if notes:
            lines.append("")
            lines.extend(f"Note: {n}" for n in notes)
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def tidy(self) -> pd.DataFrame:
        out = self.table.reset_index().rename(columns={"index": "term"})
        out = out.rename(columns={"mean": "estimate", "sd": "std_error"})
        return out

    def to_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "model": self.model,
            "formula": self.formula,
            "n_obs": int(self.n_obs),
            "sampler": self.sampler,
            "chains": int(self.chains),
            "n_draws": int(self.n_draws),
            "burnin": int(self.burnin),
            "thin": int(self.thin),
            "acceptance_rate": self.acceptance_rate,
            "level": self.level,
            "prior": {
                k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.prior.items()
            },
            "posterior": {
                str(term): {c: float(row[c]) for c in self.table.columns}
                for term, row in self.table.iterrows()
            },
            "model_info": dict(self.model_info),
            "diagnostics": {
                k: v for k, v in self.diagnostics_info.items() if k != "warnings"
            },
            "warnings": list(self.diagnostics_info.get("warnings") or []),
        }

    # -- convergence -------------------------------------------------------
    def chain_list(self) -> List[pd.DataFrame]:
        """The retained draws of each chain."""
        return [
            self.draws.loc[self.chain == c].reset_index(drop=True)
            for c in range(self.chains)
        ]

    def diagnostics(self) -> Dict[str, Any]:
        """Convergence diagnostics of the retained draws.

        Returns a dict with ``geweke``, ``heidel`` and ``raftery``
        (first chain) and, when several chains were run,
        ``gelman_rubin``. A diagnostic that cannot be computed on this
        chain (for instance Raftery-Lewis on a short run) carries the
        reason instead.
        """
        if (
            self.model in ("conjugate", "bayes_bootstrap")
            or self.model_info.get("inference") == "vb"
        ):
            raise MethodIncompatibility(
                f"model='{self.model}' here produces independent draws; "
                "there is no Markov chain to diagnose."
            )
        first = self.chain_list()[0]
        out: Dict[str, Any] = {
            "geweke": geweke_diag(first),
            "heidel": heidel_diag(first),
        }
        try:
            out["raftery"] = raftery_diag(first)
        except DataInsufficient as exc:
            out["raftery"] = str(exc)
        if self.chains > 1:
            out["gelman_rubin"] = gelman_rubin(self.chain_list())
        return out

    # -- prediction --------------------------------------------------------
    def _design(self, data: Optional[pd.DataFrame]) -> np.ndarray:
        if self._model is None:
            raise MethodIncompatibility(
                "predict() is available for regression fits, not for a "
                "user-supplied statistic."
            )
        if data is None:
            return np.asarray(self._model.X)
        # a multinomial logit repeats the design columns once per level;
        # its coefficient names are not column names
        names = getattr(self._model, "base_xnames", self._model.xnames)
        return design_for(self._design_info, names, data)

    def predict(
        self,
        data: Optional[pd.DataFrame] = None,
        what: str = "mean",
        level: Optional[float] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Posterior of the expected outcome at the rows of ``data``.

        Parameters
        ----------
        data : DataFrame, optional
            New data; the estimation sample when omitted.
        what : {'mean', 'interval', 'linear', 'draws', 'probabilities'}
            ``'mean'``: posterior mean of ``E[y | x]`` (the probability
            for logit / probit, the rate for count models, the latent
            mean for tobit, the conditional quantile for quantile
            regression, the latent index for ordered probit).
            ``'interval'``: that mean with its credible interval.
            ``'linear'``: posterior mean of the linear index.
            ``'draws'``: the full matrix, one row per posterior draw.
            ``'probabilities'``: for ordered probit, the posterior mean
            probability of each category.
        level : float, optional
            Mass of the interval for ``what='interval'``.
        """
        X = self._design(data)
        d = self.draws.to_numpy()
        if what == "probabilities":
            if self.model not in ("oprobit", "mlogit"):
                raise MethodIncompatibility(
                    "what='probabilities' is for model='oprobit' and 'mlogit'."
                )
            probs = self._model.category_probabilities(d, X)
            return pd.DataFrame(probs, columns=[str(v) for v in self._model.levels])
        eta = self._linear_index(d, X, data)
        if what == "linear":
            return np.asarray(eta.mean(axis=0))
        mu = self._model.expected_value(eta)
        if what == "mean":
            return mu.mean(axis=0)
        if what == "draws":
            return mu
        if what == "interval":
            level = self.level if level is None else level
            lo = (1.0 - level) / 2.0
            return pd.DataFrame(
                {
                    "mean": mu.mean(axis=0),
                    "lower": np.quantile(mu, lo, axis=0),
                    "upper": np.quantile(mu, 1.0 - lo, axis=0),
                }
            )
        raise MethodIncompatibility(
            "what must be 'mean', 'interval', 'linear', 'draws' or " "'probabilities'."
        )

    # -- marginal likelihood ----------------------------------------------
    def log_marginal_likelihood(
        self, method: Optional[str] = None, alpha: float = 0.05
    ) -> float:
        """Log marginal likelihood ``log p(y | model)``.

        Parameters
        ----------
        method : {'exact', 'chib', 'gelfand-dey', 'laplace'}, optional
            ``'exact'``: closed form, conjugate model only.
            ``'chib'``: Chib (1995) from the Gibbs output; normal and
            probit models.
            ``'gelfand-dey'``: Gelfand and Dey (1994) with Geweke's
            truncated normal weighting density; every model.
            ``'laplace'``: Laplace approximation at the posterior mode;
            every model with a smooth posterior (not quantile).
            Default: exact where it exists, then Chib, then Gelfand-Dey.
        alpha : float, default 0.05
            Tail mass truncated from the weighting density of the
            Gelfand-Dey estimator.

        Returns
        -------
        float

        Notes
        -----
        A marginal likelihood is only meaningful under a proper prior and
        depends on it: with a prior variance of ``c`` on a coefficient the
        marginal likelihood falls like ``c^{-1/2}`` as ``c`` grows.
        ``marginal_likelihood_details`` has the numerical standard error
        of the simulation-based estimates.
        """
        details = self.marginal_likelihood_details(method, alpha)
        return float(details["log_marginal_likelihood"])

    def marginal_likelihood_details(
        self, method: Optional[str] = None, alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Log marginal likelihood with its method and numerical error."""
        mdl = self._model
        if self.model == "bayes_bootstrap":
            raise MethodIncompatibility(
                "The Bayesian bootstrap has no parametric likelihood and "
                "therefore no marginal likelihood."
            )
        if self.model_info.get("inference") == "vb":
            raise MethodIncompatibility(
                "A variational fit has an evidence lower bound, "
                "model_info['elbo'], not a marginal likelihood."
            )
        if not hasattr(mdl, "log_kernel") and self.model != "conjugate":
            raise MethodIncompatibility(
                f"A marginal likelihood is not implemented for model="
                f"'{self.model}'."
            )
        if method is None:
            if self.model == "conjugate":
                method = "exact"
            elif getattr(mdl, "has_chib", False):
                method = "chib"
            else:
                method = "gelfand-dey"
        method = method.lower().replace("_", "-")
        key = f"{method}:{alpha}"
        if key in self._lml_cache:
            return dict(self._lml_cache[key])
        if self.model == "conjugate":
            if method != "exact":
                raise MethodIncompatibility(
                    "model='conjugate' has a closed-form marginal likelihood; "
                    "use method='exact'."
                )
            out: Dict[str, Any] = {
                "log_marginal_likelihood": mdl.log_marginal_likelihood(),
                "method": "exact",
                "mcse": 0.0,
            }
        elif method == "exact":
            raise MethodIncompatibility(
                "Only model='conjugate' has an exact marginal likelihood. Use "
                "'chib', 'gelfand-dey' or 'laplace'."
            )
        elif method == "chib":
            if not getattr(mdl, "has_chib", False):
                raise MethodIncompatibility(
                    f"Chib's method is implemented for the normal and probit "
                    f"models, not model='{self.model}'. Use 'gelfand-dey'."
                )
            d = self.draws.to_numpy()
            if self.model == "probit":
                val, info = mdl.chib(d, self._extras["cond_mean"])
            else:
                val, info = mdl.chib(d)
            out = {"log_marginal_likelihood": val, "method": "chib", **info}
        elif method == "laplace":
            mode, cov = mdl.mode()
            val = (
                mdl.log_kernel(mode)
                + 0.5 * mode.size * np.log(2.0 * np.pi)
                + 0.5 * np.linalg.slogdet(cov)[1]
            )
            out = {"log_marginal_likelihood": float(val), "method": "laplace"}
        elif method == "gelfand-dey":
            out = _gelfand_dey(mdl, self.draws.to_numpy(), alpha)
        else:
            raise MethodIncompatibility(
                "method must be 'exact', 'chib', 'gelfand-dey' or 'laplace'; "
                f"got {method!r}."
            )
        self._lml_cache[key] = dict(out)
        return out

    # -- plots -------------------------------------------------------------
    def plot(self, params: Optional[Sequence[str]] = None, kind: str = "trace") -> Any:
        """Trace and density plots of the draws.

        Parameters
        ----------
        params : list of str, optional
            Parameters to show; all by default (at most 12).
        kind : {'trace', 'density', 'acf'}
        """
        import matplotlib.pyplot as plt

        cols = list(params) if params is not None else list(self.draws.columns)[:12]
        missing = [c for c in cols if c not in self.draws.columns]
        if missing:
            raise MethodIncompatibility(
                f"Unknown parameters {missing}; available: "
                f"{list(self.draws.columns)}."
            )
        if kind not in ("trace", "density", "acf"):
            raise MethodIncompatibility("kind must be 'trace', 'density' or 'acf'.")
        ncol = 2 if kind == "trace" else 1
        fig, axes = plt.subplots(
            len(cols), ncol, figsize=(5.0 * ncol, 2.2 * len(cols)), squeeze=False
        )
        for i, c in enumerate(cols):
            if kind == "trace":
                for ch in range(self.chains):
                    x = self.draws.loc[self.chain == ch, c].to_numpy()
                    axes[i, 0].plot(x, lw=0.6)
                    axes[i, 1].hist(x, bins=40, density=True, alpha=0.6)
                axes[i, 0].set_title(f"{c}: trace")
                axes[i, 1].set_title(f"{c}: posterior")
            elif kind == "density":
                axes[i, 0].hist(self.draws[c].to_numpy(), bins=50, density=True)
                axes[i, 0].set_title(str(c))
            else:
                x = self.draws.loc[self.chain == 0, c].to_numpy()
                x = x - x.mean()
                nl = min(50, x.size - 1)
                acf = np.array(
                    [1.0] + [x[:-k] @ x[k:] / (x @ x) for k in range(1, nl + 1)]
                )
                axes[i, 0].bar(np.arange(nl + 1), acf, width=0.6)
                axes[i, 0].set_title(f"{c}: autocorrelation")
        fig.tight_layout()
        return fig


def _gelfand_dey(mdl: Any, draws: np.ndarray, alpha: float) -> Dict[str, Any]:
    """Gelfand and Dey (1994) estimator with Geweke's (1999) truncated
    normal weighting density, on the model's unconstrained scale."""
    if not 0.0 < alpha < 1.0:
        raise MethodIncompatibility(f"alpha must be in (0, 1); got {alpha}.")
    u = np.atleast_2d(mdl.to_u(draws))
    n, d = u.shape
    if n < 10 * d:
        raise DataInsufficient(
            f"The Gelfand-Dey estimator needs many more draws ({n}) than "
            f"parameters ({d})."
        )
    center = u.mean(axis=0)
    cov = np.atleast_2d(np.cov(u, rowvar=False))
    try:
        chol = linalg.cholesky(cov, lower=True)
    except linalg.LinAlgError as exc:
        raise MethodIncompatibility(
            "The posterior draws have a singular covariance matrix; the "
            "Gelfand-Dey estimator is not defined."
        ) from exc
    z = linalg.solve_triangular(chol, (u - center).T, lower=True).T
    dist = (z * z).sum(axis=1)
    inside = dist <= stats.chi2.ppf(1.0 - alpha, d)
    log_g = (
        -0.5 * d * np.log(2.0 * np.pi)
        - np.log(np.diag(chol)).sum()
        - 0.5 * dist
        - np.log(1.0 - alpha)
    )
    log_w = np.full(n, -np.inf)
    idx = np.where(inside)[0]
    kern = np.array([mdl.log_kernel(u[i]) for i in idx])
    log_w[idx] = log_g[idx] - kern
    shift = np.max(log_w[idx])
    w = np.exp(log_w - shift)  # zeros outside the truncation region
    mean_w = w.mean()
    lml = -(np.log(mean_w) + shift)
    s0 = _spectrum0_ar(w)
    mcse = float(np.sqrt(s0 / n) / mean_w) if np.isfinite(s0) else float("nan")
    return {
        "log_marginal_likelihood": float(lml),
        "method": "gelfand-dey",
        "mcse": mcse,
        "alpha": alpha,
    }


# --------------------------------------------------------------------------
# The entry point
# --------------------------------------------------------------------------


def _ordered_codes(col: pd.Series) -> Tuple[np.ndarray, List[Any]]:
    if isinstance(col.dtype, pd.CategoricalDtype):
        cat = col.cat.remove_unused_categories()
        return cat.cat.codes.to_numpy().astype(float), list(cat.cat.categories)
    levels = sorted(pd.unique(col.dropna()))
    mapping = {v: i for i, v in enumerate(levels)}
    return col.map(mapping).to_numpy(dtype=float), levels


def bayes_regress(
    formula: str,
    data: pd.DataFrame,
    model: str = "normal",
    prior_mean: Any = 0.0,
    prior_var: Any = None,
    sigma2_prior: Optional[Tuple[float, float]] = None,
    draws: int = 10000,
    burnin: int = 2000,
    thin: int = 1,
    chains: int = 1,
    seed: Optional[int] = None,
    level: float = 0.95,
    tune: Optional[float] = None,
    quantile: float = 0.5,
    scale: Optional[float] = None,
    scale_prior: Tuple[float, float] = (0.001, 0.001),
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    dof: float = 5.0,
    size_prior: Optional[Tuple[float, float]] = None,
    cut_prior_var: float = 1.0,
    inference: str = "mcmc",
    prior: str = "vague",
    offset: Any = None,
    exposure: Optional[str] = None,
) -> BayesRegressResult:
    """Bayesian regression by MCMC.

    Fits one of ten likelihoods with a normal prior on the coefficients
    and returns the posterior draws with their summary, convergence
    diagnostics and marginal likelihood.

    Parameters
    ----------
    formula : str
        ``'y ~ x1 + x2'``.
    data : DataFrame
    model : str, default 'normal'
        ``'normal'``    Gaussian linear model, independent priors
        ``beta ~ N(b0, B0)``, ``sigma2 ~ IG(alpha0/2, delta0/2)``; Gibbs.
        ``'conjugate'`` Gaussian linear model with the conjugate prior
        ``beta | sigma2 ~ N(b0, sigma2 B0)``; closed form, exact marginal
        likelihood.
        ``'t'``         linear model with Student-t errors of ``dof``
        degrees of freedom (outlier-robust); Gibbs.
        ``'logit'``, ``'poisson'``, ``'negbin'``  random-walk Metropolis
        with a proposal shaped by the posterior curvature at the mode.
        ``'probit'``    Albert and Chib (1993) data augmentation.
        ``'oprobit'``   ordered probit: Albert and Chib (1993) data
        augmentation, the cutpoints updated by a Metropolis step with the
        latent data integrated out (in the spirit of Cowles 1996).
        ``'tobit'``     censored regression (``lower=``, ``upper=``),
        Chib (1992).
        ``'quantile'``  asymmetric Laplace likelihood at ``quantile=``,
        Kozumi and Kobayashi (2011).
        ``'mlogit'``    multinomial logit, first level of the outcome as
        the base; coefficients are named ``<level>:<term>``; random-walk
        Metropolis.
    prior_mean : float or array, default 0
        Prior mean of the coefficients (scalar or one per coefficient,
        in the order of the design matrix).
    prior_var : float, array or matrix, optional
        Prior variance: a scalar, a vector of variances or a full
        covariance matrix. Default 1000 (1 for ``'conjugate'``, where it
        multiplies ``sigma2``, and for the scale-free binary and ordered
        models 100). The default is vague only for regressors on a
        moderate scale; a note is attached to the result when it is not.
    sigma2_prior : (alpha0, delta0), optional
        ``sigma2 ~ InvGamma(alpha0 / 2, delta0 / 2)`` for the normal,
        conjugate, t and tobit models. Default ``(0.001, 0.001)``; with
        ``prior='weakly_informative'`` the default is instead
        ``sigma ~ Exponential(1 / sd(y))``.
    draws : int, default 10000
        Draws kept per chain.
    burnin : int, default 2000
        Iterations discarded at the start of each chain.
    thin : int, default 1
        Keep one draw in ``thin``; each chain runs
        ``burnin + draws * thin`` iterations.
    chains : int, default 1
        Number of independent chains. With more than one, chains start
        from dispersed values and the Gelman-Rubin factor is reported.
    seed : int, optional
    level : float, default 0.95
        Mass of the credible intervals in the summary.
    tune : float, optional
        Scale of the Metropolis proposal relative to the posterior
        curvature. Default ``2.38 / sqrt(number of parameters moved)``.
    quantile : float, default 0.5
        Quantile for ``model='quantile'``.
    scale : float, optional
        Fix the scale of the asymmetric Laplace likelihood instead of
        estimating it (``scale=1`` is R ``MCMCpack::MCMCquantreg``). With
        a fixed scale the posterior spread depends on the units of ``y``.
    scale_prior : (n0, s0), default (0.001, 0.001)
        ``sigma ~ InvGamma(n0 / 2, s0 / 2)`` on the estimated scale of
        the asymmetric Laplace likelihood.
    lower, upper : float, optional
        Censoring points for ``model='tobit'``. Observations at or beyond
        a point are treated as censored there.
    dof : float, default 5
        Degrees of freedom for ``model='t'``.
    size_prior : (shape, rate), optional
        Gamma prior on the size ``1 / alpha`` of the negative binomial.
        Default ``(0.5, 0.1)``; ``(1, 1)``, an Exponential(1), with
        ``prior='weakly_informative'``.
    cut_prior_var : float, default 1
        Prior variance of the log distance between consecutive cutpoints
        of the ordered probit (mean zero).
    inference : {'mcmc', 'vb'}, default 'mcmc'
        ``'vb'`` (``model='normal'`` only): mean-field variational Bayes,
        ``q(beta) q(sigma2)`` by coordinate ascent. Fast and
        deterministic; the means are close to the posterior means, the
        standard deviations are too small when coefficients and variance
        are dependent a posteriori. ``draws`` are then independent draws
        from the approximation and ``model_info['elbo']`` is the evidence
        lower bound.
    prior : {'vague', 'weakly_informative'}, default 'vague'
        ``'vague'`` uses ``prior_mean`` and ``prior_var`` as given.
        ``'weakly_informative'`` scales independent normal priors to the
        data (models normal, logit, probit, poisson, negbin): standard
        deviation ``2.5 sd(y) / sd(x)`` for each slope and ``2.5 sd(y)``
        for the intercept, centred at ``mean(y)``, with the regressors
        centred; ``sd(y)`` is replaced by 1 and ``mean(y)`` by 0 outside
        the Gaussian model. These are the defaults of R ``rstanarm``.
        They rule out effects that are absurd on the scale of the data
        and little else, so the result does not depend on the units of
        the regressors, as it does under a fixed ``prior_var``.
        ``prior_mean`` and ``prior_var`` must be left at their defaults.
    offset : str or array, optional
        A term added to the linear index with coefficient one (logit,
        Poisson and negative binomial models): a column of ``data`` or
        an array. Predictions for new data need the column form.
    exposure : str, optional
        Column whose logarithm is the offset of a count model.

    Returns
    -------
    BayesRegressResult

    Notes
    -----
    Convergence is checked on every fit: a ``ConvergenceWarning`` is
    raised when the smallest effective sample size is below 100 or the
    split potential scale reduction factor exceeds 1.05.
    ``result.diagnostics()`` runs the Geweke, Heidelberger-Welch and
    Raftery-Lewis diagnostics.

    The result carries the predictive side of the model:
    ``posterior_linpred`` (linear index), ``posterior_epred`` (expected
    outcome) and ``posterior_predict`` (new outcomes, with observation
    noise) return one row per draw, and ``log_lik`` the pointwise
    log-likelihoods that :func:`statspai.loo`, :func:`statspai.waic` and
    :func:`statspai.kfold` consume. See also :func:`statspai.ppc`,
    :func:`statspai.bayes_r2` and :func:`statspai.loo_r2`.

    The ordered probit reports the slopes and ``J - 1`` cutpoints and has
    no intercept, as ``sp.oprobit``. The prior ``N(prior_mean, prior_var)``
    applies to the slopes and to minus the first cutpoint.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=300)})
    >>> df["y"] = (0.3 + df["x"] + rng.normal(size=300) > 0).astype(int)
    >>> fit = sp.bayes_regress("y ~ x", df, model="probit", draws=2000,
    ...                        burnin=500, seed=1)
    >>> bool(fit.prob("x > 0") > 0.99)
    True
    >>> lml = fit.log_marginal_likelihood()

    References
    ----------
    albert1993bayesian, cowles1996accelerating, chib1992bayes,
    chib1995marginal, gelfand1990sampling, gelfand1994bayesian,
    geweke1999using, kozumi2011gibbs, ramirezhassan2026introduction,
    gelman2008weakly, gelman2020regression
    """
    model_in = str(model).lower()
    model_key = _MODEL_ALIASES.get(model_in, model_in)
    if model_key not in M.MODELS:
        raise MethodIncompatibility(
            f"Unknown model {model!r}. Available: {', '.join(M.MODELS)}."
        )
    check_mcmc_args(draws, burnin, thin, chains)
    call = {
        "model": model,
        "prior_mean": prior_mean,
        "prior_var": prior_var,
        "sigma2_prior": sigma2_prior,
        "draws": draws,
        "burnin": burnin,
        "thin": thin,
        "chains": chains,
        "seed": seed,
        "level": level,
        "tune": tune,
        "quantile": quantile,
        "scale": scale,
        "scale_prior": scale_prior,
        "lower": lower,
        "upper": upper,
        "dof": dof,
        "size_prior": size_prior,
        "cut_prior_var": cut_prior_var,
        "inference": inference,
        "prior": prior,
        "offset": offset,
        "exposure": exposure,
    }
    prior_kind = str(prior).lower().replace("-", "_")
    if prior_kind in ("weakly_informative", "weak", "auto", "rstanarm"):
        prior_kind = "weakly_informative"
    elif prior_kind not in ("vague", "default"):
        raise MethodIncompatibility(
            f"prior must be 'vague' or 'weakly_informative'; got {prior!r}."
        )
    else:
        prior_kind = "vague"
    if not 0.0 < level < 1.0:
        raise MethodIncompatibility(f"level must be in (0, 1); got {level}.")
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility("data must be a pandas DataFrame.")
    if "~" not in formula:
        raise MethodIncompatibility(
            f"formula must look like 'y ~ x1 + x2'; got {formula!r}."
        )

    levels: List[Any] = []
    work = data
    if model_key in ("oprobit", "mlogit"):
        lhs = formula.split("~", 1)[0].strip()
        if lhs not in data.columns:
            raise MethodIncompatibility(
                f"For model='{model_key}' the left-hand side must be a column "
                f"of data; {lhs!r} is not."
            )
        work = data.loc[data[lhs].notna()].copy()
        codes, levels = _ordered_codes(work[lhs])
        work[lhs] = codes
        rhs = formula.split("~", 1)[1]
        no_const = "-1" in rhs.replace(" ", "") or "+0" in rhs.replace(" ", "")
        if model_key == "oprobit" and no_const:
            raise MethodIncompatibility(
                "The ordered probit has free cutpoints in place of an "
                "intercept; write the formula without '- 1'."
            )
    y_df, X_df = create_design_matrices(formula, work)
    y = np.asarray(y_df, dtype=float).reshape(-1)
    X = np.asarray(X_df, dtype=float)
    xnames = [str(c) for c in X_df.columns]
    design_info = getattr(X_df, "design_info", None)
    n, k = X.shape
    if n <= k:
        raise DataInsufficient(
            f"{n} observations for {k} coefficients; the model needs more "
            "observations than coefficients."
        )
    rank = np.linalg.matrix_rank(X)
    if rank < k:
        raise MethodIncompatibility(
            f"The regressors are collinear (rank {rank} of {k}). Drop the "
            "redundant columns; a prior does not make the dropped direction "
            "meaningful."
        )

    frame = work.loc[y_df.index] if hasattr(y_df, "index") else work
    offset_spec = _OffsetSpec.build(offset, exposure, work, frame, model_key)

    default_prior = prior_var is None and prior_kind == "vague"
    weak_info: Dict[str, Any] = {}
    exp_sigma_rate: Optional[float] = None
    if prior_kind == "weakly_informative":
        if prior_var is not None or np.any(np.asarray(prior_mean) != 0.0):
            raise MethodIncompatibility(
                "prior='weakly_informative' sets the coefficient prior from "
                "the data; leave prior_mean and prior_var at their defaults, "
                "or use prior='vague' with your own."
            )
        if inference != "mcmc":
            raise MethodIncompatibility(
                "prior='weakly_informative' is available with inference='mcmc'."
            )
        prior_mean, prior_var, weak_info = W.weakly_informative_prior(
            model_key, y, X, xnames
        )
        if model_key == "normal" and sigma2_prior is None:
            exp_sigma_rate = 1.0 / float(np.std(y, ddof=1))
        if size_prior is None:
            size_prior = (1.0, 1.0)
    if prior_var is None:
        prior_var = {
            "probit": 100.0,
            "logit": 100.0,
            "oprobit": 100.0,
            "mlogit": 100.0,
        }.get(model_key, 1000.0)
    a0, d0 = (float(v) for v in (sigma2_prior or (0.001, 0.001)))
    if size_prior is None:
        size_prior = (0.5, 0.1)

    prior_info: Dict[str, Any] = {
        "coefficients": "normal",
        "prior_mean": prior_mean,
        "prior_var": prior_var,
    }
    if weak_info:
        prior_info.update(weak_info)
        prior_info["kind"] = "weakly_informative"
    info: Dict[str, Any] = {}

    if model_key == "conjugate":
        conj = _ConjugateNormal(y, X, xnames, prior_mean, prior_var, a0, d0)
        if conj.an <= 4:
            raise DataInsufficient(
                "The conjugate posterior needs alpha0 + n > 4 for finite " "variances."
            )
        rng = spawn_rngs(seed, 1)[0]
        arr = conj.sample(rng, draws)
        d_df = pd.DataFrame(arr, columns=conj.names)
        table = conj.exact_table(level)
        table.insert(2, "mcse", 0.0)
        table.insert(3, "ess", float(draws))
        prior_info.update(
            {
                "coefficients": "normal, scaled by sigma2",
                "sigma2": f"InvGamma({a0 / 2:g}, {d0 / 2:g})",
            }
        )
        info = {
            "posterior_mean": conj.bn,
            "posterior_scale": conj.dn / conj.an * conj.Bn,
            "posterior_df": conj.an,
            "sigma2_shape": conj.an / 2.0,
            "sigma2_rate": conj.dn / 2.0,
        }
        res = BayesRegressResult(
            model="conjugate",
            formula=formula,
            params=table["mean"].copy(),
            std_errors=table["sd"].copy(),
            table=table,
            draws=d_df,
            chain=np.zeros(draws, dtype=int),
            n_obs=n,
            n_draws=draws,
            burnin=0,
            thin=1,
            chains=1,
            sampler=conj.sampler,
            acceptance_rate=None,
            prior=prior_info,
            level=level,
            model_info=info,
            diagnostics_info={"warnings": []},
            _model=conj,
            _design_info=design_info,
            _frame=frame,
            _call=call,
        )
        _note_default_prior(
            res, default_prior, conj.b0, conj.B0 * table.loc["sigma2", "mean"]
        )
        return res

    inference = str(inference).lower()
    if inference not in ("mcmc", "vb"):
        raise MethodIncompatibility("inference must be 'mcmc' or 'vb'.")
    if inference == "vb" and model_key != "normal":
        raise MethodIncompatibility(
            "inference='vb' is implemented for model='normal' only."
        )
    if model_key == "normal" and exp_sigma_rate is not None:
        mdl: Any = W.NormalExpSigmaModel(
            y, X, xnames, prior_mean, prior_var, exp_sigma_rate
        )
        prior_info["sigma"] = f"Exponential(rate {exp_sigma_rate:.5g})"
    elif model_key == "normal":
        mdl = M.NormalModel(y, X, xnames, prior_mean, prior_var, a0, d0)
        prior_info["sigma2"] = f"InvGamma({a0 / 2:g}, {d0 / 2:g})"
        if inference == "vb":
            res = _vb_normal(mdl, formula, draws, seed, level, prior_info, design_info)
            _note_default_prior(res, default_prior, mdl.b0, mdl.B0)
            return res
    elif model_key == "t":
        mdl = M.StudentTModel(y, X, xnames, prior_mean, prior_var, a0, d0, dof)
        prior_info["sigma2"] = f"InvGamma({a0 / 2:g}, {d0 / 2:g})"
        info["dof"] = float(dof)
    elif model_key == "probit":
        mdl = M.ProbitModel(y, X, xnames, prior_mean, prior_var)
    elif model_key == "logit":
        mdl = M.LogitModel(y, X, xnames, prior_mean, prior_var, tune)
    elif model_key == "poisson":
        mdl = M.PoissonModel(y, X, xnames, prior_mean, prior_var, tune)
    elif model_key == "negbin":
        sh, rt = (float(v) for v in size_prior)
        mdl = M.NegBinModel(y, X, xnames, prior_mean, prior_var, tune, sh, rt)
        prior_info["size"] = f"Gamma({sh:g}, rate {rt:g}) on 1 / alpha"
    elif model_key == "tobit":
        mdl = M.TobitModel(y, X, xnames, prior_mean, prior_var, a0, d0, lower, upper)
        prior_info["sigma2"] = f"InvGamma({a0 / 2:g}, {d0 / 2:g})"
        info.update(
            {
                "lower": mdl.lower,
                "upper": mdl.upper,
                "n_left_censored": int(mdl.left.sum()),
                "n_right_censored": int(mdl.right.sum()),
            }
        )
    elif model_key == "quantile":
        n0, s0 = (float(v) for v in scale_prior)
        mdl = M.QuantileModel(
            y, X, xnames, prior_mean, prior_var, quantile, n0, s0, scale
        )
        info["quantile"] = float(quantile)
        if scale is None:
            prior_info["sigma"] = f"InvGamma({n0 / 2:g}, {s0 / 2:g})"
        else:
            info["scale"] = float(scale)
    elif model_key == "mlogit":
        mdl = M.MultinomialLogitModel(y, X, xnames, prior_mean, prior_var, tune, levels)
        info["levels"] = [str(v) for v in levels]
        info["base_level"] = str(levels[0])
    else:  # oprobit
        if "Intercept" not in xnames:
            raise MethodIncompatibility(
                "The ordered probit needs the default intercept term in the "
                "formula (it is absorbed into the cutpoints)."
            )
        order = [xnames.index("Intercept")] + [
            i for i, c in enumerate(xnames) if c != "Intercept"
        ]
        if order != list(range(k)):
            X = X[:, order]
            xnames = [xnames[i] for i in order]
        mdl = M.OrderedProbitModel(
            y, X, xnames, prior_mean, prior_var, tune, cut_prior_var, levels
        )
        prior_info["cutpoints"] = (
            f"log increments ~ N(0, {cut_prior_var:g}); "
            "minus the first cutpoint shares the coefficient prior"
        )
        info["levels"] = [str(v) for v in levels]

    if offset_spec is not None:
        mdl.offset = offset_spec.values
        info["offset"] = offset_spec.label

    n_iter = burnin + draws * thin
    rngs = spawn_rngs(seed, chains)
    pieces: List[np.ndarray] = []
    accepts: List[float] = []
    extras: Dict[str, Any] = {}
    for c in range(chains):
        out = mdl.sample(rngs[c], n_iter, jitter=chains > 1)
        keep = slice(burnin, n_iter, thin)
        pieces.append(out["draws"][keep])
        if out["accept"] is not None:
            accepts.append(out["accept"])
        for key, val in out["extras"].items():
            if isinstance(val, np.ndarray) and val.shape[0] == n_iter:
                extras.setdefault(key, []).append(val[keep])
            else:
                extras[key] = val
    arr = np.vstack(pieces)
    for key, val in list(extras.items()):
        if isinstance(val, list):
            extras[key] = np.vstack(val)
    if not np.isfinite(arr).all():
        raise MethodIncompatibility(
            "The sampler produced non-finite draws. This usually means the "
            "likelihood is degenerate for these data (separation, an "
            "outcome outside the model's support) or the prior is improper."
        )
    names = mdl.names
    d_df = pd.DataFrame(arr, columns=names)
    chain_idx = np.repeat(np.arange(chains), draws)

    summ = mcmc_summary(d_df, quantiles=())
    if chains > 1:
        # pool the per-chain effective sample sizes
        ess = np.zeros(len(names))
        for c in range(chains):
            ess += mcmc_summary(d_df.loc[chain_idx == c], quantiles=())[
                "ess"
            ].to_numpy()
        summ["ess"] = ess
        with np.errstate(divide="ignore", invalid="ignore"):
            summ["ts_se"] = summ["sd"] / np.sqrt(ess)
    lo = (1.0 - level) / 2.0
    table = pd.DataFrame(
        {
            "mean": summ["mean"],
            "sd": summ["sd"],
            "mcse": summ["ts_se"],
            "ess": summ["ess"],
            "lower": d_df.quantile(lo),
            "median": d_df.quantile(0.5),
            "upper": d_df.quantile(1.0 - lo),
            "prob_positive": (d_df > 0).mean(),
        }
    )

    diag_info: Dict[str, Any] = {"warnings": []}
    diag_info["min_ess"] = float(table["ess"].min())
    try:
        gr = gelman_rubin([d_df.loc[chain_idx == c] for c in range(chains)], split=True)
        diag_info["max_split_rhat"] = float(gr.table["psrf"].max())
    except (MethodIncompatibility, DataInsufficient):
        diag_info["max_split_rhat"] = float("nan")

    res = BayesRegressResult(
        model=model_key,
        formula=formula,
        params=table["mean"].copy(),
        std_errors=table["sd"].copy(),
        table=table,
        draws=d_df,
        chain=chain_idx,
        n_obs=n,
        n_draws=draws * chains,
        burnin=burnin,
        thin=thin,
        chains=chains,
        sampler=mdl.sampler,
        acceptance_rate=float(np.mean(accepts)) if accepts else None,
        prior=prior_info,
        level=level,
        model_info=info,
        diagnostics_info=diag_info,
        _model=mdl,
        _extras=extras,
        _design_info=design_info,
        _frame=frame,
        _call=call,
        _offset=offset_spec,
    )

    msgs = []
    if diag_info["min_ess"] < 100:
        worst = str(table["ess"].idxmin())
        msgs.append(
            f"effective sample size is {diag_info['min_ess']:.0f} for " f"'{worst}'"
        )
    rhat = diag_info["max_split_rhat"]
    if np.isfinite(rhat) and rhat > 1.05:
        msgs.append(f"split potential scale reduction factor is {rhat:.3f}")
    if res.acceptance_rate is not None and not 0.1 <= res.acceptance_rate <= 0.7:
        msgs.append(
            f"Metropolis acceptance rate is {res.acceptance_rate:.2f} "
            "(aim for 0.2 to 0.5; change tune=)"
        )
    if msgs:
        text = (
            "The chain may not have converged or mixes slowly: "
            + "; ".join(msgs)
            + ". Increase draws / burnin, thin the chain, or check "
            "result.diagnostics()."
        )
        diag_info["warnings"].append(text)
        warnings.warn(text, ConvergenceWarning, stacklevel=2)
    _note_default_prior(res, default_prior, mdl.b0, mdl.B0)
    return res


def _vb_normal(
    mdl: Any,
    formula: str,
    draws: int,
    seed: Optional[int],
    level: float,
    prior: Dict[str, Any],
    design_info: Any,
) -> BayesRegressResult:
    """Mean-field variational Bayes for the normal linear model.

    ``q(beta) = N(m, S)``, ``q(sigma2) = InvGamma(an / 2, dn / 2)`` with
    the coordinate ascent updates
    ``S = (B0^{-1} + r X'X)^{-1}``, ``m = S (B0^{-1} b0 + r X'y)``,
    ``dn = d0 + |y - X m|^2 + tr(X'X S)``, ``r = E_q[1 / sigma2] = an / dn``.
    """
    n, k = mdl.n, mdl.k
    an = mdl.a0 + n
    r = 1.0 / max(mdl._ssr(mdl._ols()) / max(n - k, 1), 1e-12)
    logdet_b0 = np.linalg.slogdet(mdl.B0)[1]
    elbo_old = -np.inf
    elbo = -np.inf
    m = np.zeros(k)
    S = np.eye(k)
    dn = 1.0
    for it in range(500):
        S = linalg.inv(mdl.B0inv + r * mdl.XtX)
        S = 0.5 * (S + S.T)
        m = S @ (mdl.B0inv_b0 + r * mdl.Xty)
        dn = mdl.d0 + mdl._ssr(m) + float(np.trace(mdl.XtX @ S))
        r_prev, r = r, an / dn
        e_log_s2 = np.log(dn / 2.0) - special.digamma(an / 2.0)
        dev = m - mdl.b0
        elbo = float(
            -0.5 * n * (np.log(2 * np.pi) + e_log_s2)
            - 0.5 * r * (mdl._ssr(m) + np.trace(mdl.XtX @ S))
            - 0.5 * (k * np.log(2 * np.pi) + logdet_b0)
            - 0.5 * (dev @ mdl.B0inv @ dev + np.trace(mdl.B0inv @ S))
            + 0.5 * mdl.a0 * np.log(mdl.d0 / 2.0)
            - special.gammaln(mdl.a0 / 2.0)
            - (mdl.a0 / 2.0 + 1.0) * e_log_s2
            - 0.5 * mdl.d0 * r
            + 0.5 * (k * (1 + np.log(2 * np.pi)) + np.linalg.slogdet(S)[1])
            + an / 2.0
            + np.log(dn / 2.0)
            + special.gammaln(an / 2.0)
            - (1.0 + an / 2.0) * special.digamma(an / 2.0)
        )
        if (
            abs(elbo - elbo_old) < 1e-12 * (1.0 + abs(elbo))
            and abs(r - r_prev) < 1e-13 * r
        ):
            break
        elbo_old = elbo
    rng = spawn_rngs(seed, 1)[0]
    beta = rng.multivariate_normal(m, S, size=draws, method="cholesky")
    s2 = (dn / 2.0) / rng.gamma(an / 2.0, size=draws)
    d_df = pd.DataFrame(np.column_stack([beta, s2]), columns=mdl.names)
    lo = (1.0 - level) / 2.0
    sd_beta = np.sqrt(np.diag(S))
    ig = stats.invgamma(an / 2.0, scale=dn / 2.0)
    mean = np.append(m, dn / (an - 2.0))
    sd = np.append(sd_beta, float(ig.std()))
    zq = stats.norm.ppf(1.0 - lo)
    table = pd.DataFrame(
        {
            "mean": mean,
            "sd": sd,
            "mcse": 0.0,
            "ess": float(draws),
            "lower": np.append(m - zq * sd_beta, ig.ppf(lo)),
            "median": np.append(m, ig.ppf(0.5)),
            "upper": np.append(m + zq * sd_beta, ig.ppf(1.0 - lo)),
            "prob_positive": np.append(stats.norm.cdf(m / sd_beta), 1.0),
        },
        index=mdl.names,
    )
    return BayesRegressResult(
        model="normal",
        formula=formula,
        params=table["mean"].copy(),
        std_errors=table["sd"].copy(),
        table=table,
        draws=d_df,
        chain=np.zeros(draws, dtype=int),
        n_obs=n,
        n_draws=draws,
        burnin=0,
        thin=1,
        chains=1,
        sampler="variational Bayes (mean field, coordinate ascent)",
        acceptance_rate=None,
        prior=prior,
        level=level,
        model_info={
            "inference": "vb",
            "elbo": elbo,
            "iterations": it + 1,
            "q_beta_mean": m,
            "q_beta_cov": S,
            "q_sigma2_shape": an / 2.0,
            "q_sigma2_rate": dn / 2.0,
        },
        diagnostics_info={"warnings": []},
        _model=mdl,
        _design_info=design_info,
    )


def _note_default_prior(
    res: BayesRegressResult, default_prior: bool, b0: np.ndarray, B0: np.ndarray
) -> None:
    """Flag a default prior that is not vague at the scale of the data."""
    if not default_prior:
        return
    k = b0.size
    names = list(res.table.index)[:k]
    if res.model == "oprobit":
        # prior is on (intercept, slopes); reported are slopes then cuts
        post_mean = np.append(
            -res.table.loc["cut1", "mean"], res.table["mean"].to_numpy()[: k - 1]
        )
        post_sd = np.append(
            res.table.loc["cut1", "sd"], res.table["sd"].to_numpy()[: k - 1]
        )
        names = ["-cut1"] + names[: k - 1]
    else:
        post_mean = res.table["mean"].to_numpy()[:k]
        post_sd = res.table["sd"].to_numpy()[:k]
    prior_sd = np.sqrt(np.diag(B0))
    far = np.abs(post_mean - b0) / prior_sd > 1.0
    tight = post_sd / prior_sd > 0.3
    bad = [n for n, f, t in zip(names, far, tight) if f or t]
    if bad:
        text = (
            "The default prior is not vague for " + ", ".join(map(str, bad)) + ": "
            "its standard deviation is of the same order as the "
            "coefficient or its posterior uncertainty, so it pulls the "
            "estimate toward zero. Use prior='weakly_informative' (priors "
            "scaled to the data), set prior_var= (and prior_mean=) to "
            "match the scale of these regressors, or rescale them."
        )
        res.diagnostics_info["warnings"].append(text)
        warnings.warn(text, StatsPAIWarning, stacklevel=3)
