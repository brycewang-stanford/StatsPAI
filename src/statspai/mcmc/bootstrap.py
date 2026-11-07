"""
The Bayesian bootstrap (Rubin 1981): ``sp.bayes_bootstrap``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.utils import create_design_matrices
from ..exceptions import DataInsufficient, MethodIncompatibility
from ..output._lineage import records_provenance
from ._chains import ChainRun
from ._core import check_mcmc_args
from .regress import BayesRegressResult


class _BootModel:
    """Just enough of the model interface for predict()."""

    name = "bayes_bootstrap"
    sampler = "Bayesian bootstrap (independent draws)"

    def __init__(self, X: np.ndarray, xnames: List[str]):
        self.X = X
        self.k = X.shape[1]
        self.xnames = xnames

    def linear_predictor(self, draws: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.asarray(draws[:, : self.k] @ X.T)

    @staticmethod
    def expected_value(eta: np.ndarray) -> np.ndarray:
        return eta


REFUSED = (
    "the Bayesian bootstrap draws independent weight vectors, so there is "
    "no chain to start, continue or burn in."
)


@records_provenance("sp.bayes_bootstrap", record_defaults=("seed",))
def bayes_bootstrap(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    statistic: Optional[Callable[[pd.DataFrame, np.ndarray], Any]] = None,
    draws: int = 2000,
    concentration: float = 1.0,
    seed: Optional[int] = None,
    level: float = 0.95,
    inits: Any = None,
    continue_from: Any = None,
    verbose: bool = False,
    keep_burnin: bool = False,
) -> BayesRegressResult:
    """Bayesian bootstrap of a regression or of any weighted statistic.

    Rubin's (1981) analogue of the bootstrap: instead of resampling rows,
    draw a weight vector from a flat Dirichlet distribution and recompute
    the statistic with those weights. The draws are a sample from the
    posterior of the statistic under a nonparametric model that puts all
    its mass on the observed rows, so no likelihood is assumed. For least
    squares the posterior standard deviations are close to
    heteroskedasticity-robust standard errors.

    Parameters
    ----------
    formula : str, optional
        ``'y ~ x1 + x2'``: weighted least squares coefficients are the
        statistic.
    data : DataFrame
        With a formula, rows with a missing value in one of its variables are
        dropped (listwise deletion), with an ``AssumptionWarning`` that counts
        them. With ``statistic=`` the rows are passed as given, and missing
        values are the statistic's to handle.
    statistic : callable, optional
        Instead of a formula, ``statistic(data, weights)`` returning a
        number, a sequence or a ``dict`` / ``Series`` of named values.
        ``weights`` sum to one.
    draws : int, default 2000
        Number of posterior draws.
    concentration : float, default 1
        Dirichlet concentration of the weights. 1 is Rubin's Bayesian
        bootstrap; larger values pull the weights toward equality.
    seed : int, optional
    level : float, default 0.95
        Mass of the credible intervals.
    inits, continue_from, keep_burnin
        Not available: the Bayesian bootstrap draws independent weight
        vectors, so there is no chain to start, continue or burn in.
        Passing any of them raises ``MethodIncompatibility``; raise
        ``draws`` for more draws.
    verbose : bool, default False
        Print a line to standard output when the weight draws start and
        end. Warnings are not affected.

    Returns
    -------
    BayesRegressResult
        With ``model='bayes_bootstrap'``. The draws are independent, so
        no convergence diagnostic applies, and there is no marginal
        likelihood.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=200)})
    >>> df["y"] = 1 + df["x"] + rng.normal(size=200)
    >>> fit = sp.bayes_bootstrap("y ~ x", df, draws=500, seed=1)
    >>> list(fit.params.index)
    ['Intercept', 'x']
    >>> med = sp.bayes_bootstrap(
    ...     data=df, statistic=lambda d, w: np.average(d["y"], weights=w),
    ...     draws=500, seed=1)

    References
    ----------
    rubin1981bayesian
    """
    check_mcmc_args(draws, 0, 1)
    ctl = ChainRun(
        "bayes_bootstrap",
        inits=inits,
        continue_from=continue_from,
        verbose=verbose,
        keep_burnin=keep_burnin,
        refuse=REFUSED,
    )
    if draws < 2:
        raise MethodIncompatibility(
            "draws must be at least 2: one weight draw has no spread to " "summarise."
        )
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            "sp.bayes_bootstrap: "
            f"data must be a pandas DataFrame; got {type(data).__name__}."
        )
    if (formula is None) == (statistic is None):
        raise MethodIncompatibility(
            "Pass either a formula (least squares) or statistic=, not both."
        )
    if not concentration > 0:
        raise MethodIncompatibility("concentration must be positive.")
    if not 0.0 < level < 1.0:
        raise MethodIncompatibility(f"level must be in (0, 1); got {level}.")
    rng = np.random.default_rng(seed)
    ctl.progress.say(f"{draws} weight draws")
    model: Any = None
    design_info = None
    if formula is not None:
        y_df, X_df = create_design_matrices(formula, data)
        y = np.asarray(y_df, dtype=float).reshape(-1)
        X = np.asarray(X_df, dtype=float)
        names = [str(c) for c in X_df.columns]
        design_info = getattr(X_df, "design_info", None)
        n, k = X.shape
        if n <= k:
            raise DataInsufficient(f"{n} observations for {k} coefficients.")
        if np.linalg.matrix_rank(X) < k:
            raise MethodIncompatibility(
                "sp.bayes_bootstrap: The regressors are collinear."
            )
        w = rng.dirichlet(np.full(n, concentration), size=draws)
        out = np.empty((draws, k))
        for s in range(draws):
            Xw = X * w[s][:, None]
            out[s] = np.linalg.solve(Xw.T @ X, Xw.T @ y)
        model = _BootModel(X, names)
    else:
        n = data.shape[0]
        if n < 2:
            raise DataInsufficient("The Bayesian bootstrap needs at least two rows.")
        w = rng.dirichlet(np.full(n, concentration), size=draws)
        first = statistic(data, w[0])  # type: ignore[misc]
        if isinstance(first, (dict, pd.Series)):
            names = [str(key) for key in dict(first).keys()]
        else:
            size = np.atleast_1d(np.asarray(first, dtype=float)).size
            names = (
                ["statistic"] if size == 1 else [f"stat{i + 1}" for i in range(size)]
            )
        out = np.empty((draws, len(names)))

        def as_row(val: Any) -> np.ndarray:
            if isinstance(val, (dict, pd.Series)):
                row = np.asarray(list(dict(val).values()), dtype=float)
            else:
                row = np.atleast_1d(np.asarray(val, dtype=float)).reshape(-1)
            if row.size != len(names):
                raise MethodIncompatibility(
                    f"statistic returned {row.size} values for one weight "
                    f"draw and {len(names)} for the first; it must return "
                    "the same quantities every time."
                )
            return row

        out[0] = as_row(first)
        for s in range(1, draws):
            out[s] = as_row(statistic(data, w[s]))  # type: ignore[misc]
        formula = "statistic(data, weights)"
    if not np.isfinite(out).all():
        raise MethodIncompatibility(
            "The statistic returned missing or infinite values for some "
            "weight draws."
        )
    d_df = pd.DataFrame(out, columns=names)
    lo = (1.0 - level) / 2.0
    sd = d_df.std(ddof=1)
    table = pd.DataFrame(
        {
            "mean": d_df.mean(),
            "sd": sd,
            "mcse": sd / np.sqrt(draws),
            "ess": float(draws),
            "lower": d_df.quantile(lo),
            "median": d_df.quantile(0.5),
            "upper": d_df.quantile(1.0 - lo),
            "prob_positive": (d_df > 0).mean(),
        }
    )
    info: Dict[str, Any] = {"concentration": float(concentration)}
    ctl.progress.say(f"{draws} draws done; the draws are independent")
    return BayesRegressResult(
        model="bayes_bootstrap",
        formula=formula,
        params=table["mean"].copy(),
        std_errors=table["sd"].copy(),
        table=table,
        draws=d_df,
        chain=np.zeros(draws, dtype=int),
        n_obs=int(n),
        n_draws=draws,
        burnin=0,
        thin=1,
        chains=1,
        sampler=_BootModel.sampler,
        acceptance_rate=None,
        prior={"weights": f"Dirichlet({concentration:g}, ..., {concentration:g})"},
        level=level,
        model_info=info,
        diagnostics_info={"warnings": []},
        _model=model,
        _design_info=design_info,
    )
