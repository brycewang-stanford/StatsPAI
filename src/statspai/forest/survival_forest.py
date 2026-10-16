"""Survival forest for right-censored outcomes (``grf::survival_forest``).

Trees split on the two-sample log-rank statistic, with each child holding
at least ``max(1, alpha * n_parent)`` failures (Ishwaran, Kogalur,
Blackstone and Lauer 2008); leaves are honest.  The conditional survival
function ``S(t | x) = P[T > t | X = x]`` is the Kaplan-Meier (or
Nelson-Aalen) estimator under the forest weights, on the grid of observed
failure times; an event time is rounded down to the last grid time at or
below it, so a coarser ``failure_times`` grid trades resolution for speed.

The forest is also the censoring and survival nuisance learner of
:func:`statspai.causal_survival_forest`.

References
----------
[@ishwaran2008random], [@athey2019generalized], [@cui2023estimating]
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from ..exceptions import DataInsufficient, MethodIncompatibility
from . import _grf_engine as engine
from ._grf_family import (
    ForestOptions,
    GRFFamilyForest,
    cluster_codes,
    resolve_inputs,
    sample_weights,
)


def failure_grid(Y: np.ndarray, D: np.ndarray, failure_times: Any = None) -> np.ndarray:
    if failure_times is None:
        grid = np.unique(Y[D > 0.5])
    else:
        grid = np.asarray(failure_times, dtype=float).ravel()
        if grid.size == 0 or not np.all(np.isfinite(grid)):
            raise MethodIncompatibility(
                "survival_forest: failure_times must be finite and non-empty.",
                recovery_hint="Pass an increasing vector of time points.",
            )
        if np.any(np.diff(grid) <= 0):
            raise MethodIncompatibility(
                "survival_forest: failure_times must be strictly increasing.",
                recovery_hint="Sort and de-duplicate the time grid.",
            )
    return np.asarray(grid, dtype=float)


def time_index(Y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Number of grid times at or below each ``Y`` (0 = before the grid)."""
    return np.searchsorted(grid, Y, side="right").astype(np.int64)


def train_survival_engine(
    X: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    grid: np.ndarray,
    *,
    num_trees: int,
    min_node_size: int,
    mtry: Optional[int],
    sample_weight: Optional[np.ndarray],
    common: Dict[str, Any],
    max_depth: Optional[int] = None,
) -> Any:
    """Train a survival forest on the StatsPAI engine."""
    tidx = time_index(Y, grid)
    ev = np.asarray(D, dtype=float)
    kw = dict(common)
    kw.pop("sample_weight", None)
    return engine.train_forest(
        X,
        np.zeros(Y.size),
        kind=engine.KIND_SURVIVAL,
        num_trees=int(num_trees),
        mtry=mtry,
        min_node_size=int(min_node_size),
        ci_group_size=1,
        max_depth=max_depth,
        M=np.column_stack([tidx.astype(float), ev]),
        params=np.array([float(grid.size)]),
        aux={
            "time_index": tidx,
            "event": ev,
            "sample_weight": (
                np.ones(Y.size)
                if sample_weight is None
                else np.asarray(sample_weight, dtype=float)
            ),
            "failure_times": np.asarray(grid, dtype=float),
        },
        sample_weight=None,
        **kw,
    )


class SurvivalForestResult(GRFFamilyForest):
    """A fitted survival forest.

    Attributes
    ----------
    failure_times : np.ndarray
        Grid on which curves are reported.
    predictions : np.ndarray
        Out-of-bag survival curves of the training rows, ``(n, n_times)``.
    prediction_type : str
        ``"Kaplan-Meier"`` or ``"Nelson-Aalen"``.

    Examples
    --------
    >>> import numpy as np, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 2)); T = rng.exponential(size=300)
    >>> sf = sp.survival_forest(time=T, event=np.ones(300), covariates=X,
    ...                         n_estimators=100)
    >>> sf.prediction_type
    'Kaplan-Meier'
    """

    _citation_keys = ("ishwaran2008random", "athey2019generalized")
    _context = "survival_forest"

    def __init__(self) -> None:
        self.failure_times = np.zeros(0)
        self.predictions = np.zeros((0, 0))
        self.prediction_type = "Kaplan-Meier"
        self.feature_names: list = []
        self.n_obs = 0
        self.detail: Dict[str, Any] = {}

    def predict(
        self,
        newdata: Any = None,
        failure_times: Any = None,
        prediction_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Survival curves at ``newdata`` (OOB for the training rows when
        None), evaluated on ``failure_times`` (default: the fit grid; other
        times take the curve's value at the last grid time at or below)."""
        ptype = self.prediction_type if prediction_type is None else prediction_type
        na = _prediction_type(ptype) == "Nelson-Aalen"
        if newdata is None:
            if (
                prediction_type is None
                or _prediction_type(ptype) == self.prediction_type
            ):
                curves = self.predictions
            else:
                curves = self._engine.predict_survival(
                    self._X, oob=True, nelson_aalen=na
                )
        else:
            curves = self._engine.predict_survival(
                self._new_X(newdata), oob=False, nelson_aalen=na
            )
        grid = self.failure_times
        if failure_times is None:
            return pd.DataFrame(curves, columns=[f"{t:g}" for t in grid])
        times = np.asarray(failure_times, dtype=float).ravel()
        vals = step_eval(curves, grid, times)
        return pd.DataFrame(vals, columns=[f"{t:g}" for t in times])

    def summary(self) -> str:
        return (
            "Survival forest (GRF engine, log-rank splits)\n"
            f"  N              : {self.n_obs}\n"
            f"  trees          : {self.num_trees}\n"
            f"  failure times  : {self.failure_times.size}\n"
            f"  estimator      : forest-weighted {self.prediction_type}"
        )

    def __repr__(self) -> str:
        return f"SurvivalForestResult(n={self.n_obs}, times={self.failure_times.size})"


def step_eval(curves: np.ndarray, grid: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Right-continuous step curves (value 1 before the grid) at ``times``."""
    k = np.searchsorted(grid, times, side="right")
    padded = np.column_stack([np.ones(curves.shape[0]), curves])
    return padded[:, k]


def _prediction_type(value: str) -> str:
    v = str(value).lower().replace("_", "-").replace(" ", "-")
    if v in ("kaplan-meier", "km"):
        return "Kaplan-Meier"
    if v in ("nelson-aalen", "na"):
        return "Nelson-Aalen"
    raise MethodIncompatibility(
        f"survival_forest: unknown prediction_type {value!r}.",
        recovery_hint="Use 'Kaplan-Meier' or 'Nelson-Aalen'.",
    )


@accepts_aliases(_strict=True, n_trees="n_estimators", seed="random_state")
def survival_forest(
    data: Optional[pd.DataFrame] = None,
    time: Any = None,
    event: Any = None,
    covariates: Any = None,
    *,
    failure_times: Any = None,
    prediction_type: str = "Kaplan-Meier",
    clusters: Any = None,
    weights: Any = None,
    equalize_cluster_weights: bool = False,
    n_estimators: int = 1000,
    min_samples_leaf: int = 15,
    max_samples: float = 0.5,
    mtry: Optional[int] = None,
    honest: bool = True,
    honesty_fraction: float = 0.5,
    honesty_prune_leaves: bool = True,
    split_alpha: float = 0.05,
    max_depth: Optional[int] = None,
    random_state: Optional[int] = 42,
    n_jobs: int = 1,
) -> SurvivalForestResult:
    """
    Survival forest: conditional survival curves ``S(t | x)`` from
    right-censored data (``grf::survival_forest``).

    Parameters
    ----------
    data : pd.DataFrame, optional
        Input data; when omitted, the other inputs are arrays.
    time : str or array-like
        Observed time (event or censoring), non-negative.
    event : str or array-like
        1 = event observed, 0 = censored.
    covariates : str, list of str or 2-D array
        Covariates ``X``.
    failure_times : array-like, optional
        Increasing grid of times for the curves; default all distinct event
        times.
    prediction_type : {"Kaplan-Meier", "Nelson-Aalen"}
    clusters, weights, equalize_cluster_weights
        Cluster-sampled trees; sample weights enter the curve estimates
        (not the log-rank splits, as in grf).
    n_estimators : int, default 1000
    min_samples_leaf : int, default 15
    max_samples, mtry, honest, honesty_fraction, honesty_prune_leaves
        Forest options (grf names ``sample.fraction``, ...).
    split_alpha : float, default 0.05
        Each child must hold at least ``max(1, split_alpha * n_parent)``
        failures (grf ``alpha``).
    max_depth, random_state, n_jobs

    Returns
    -------
    SurvivalForestResult
        ``.predictions`` (OOB curves), ``.failure_times``,
        ``.predict(newdata, failure_times=...)``.

    Examples
    --------
    >>> import numpy as np, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 2))
    >>> T = rng.exponential(1 / np.exp(0.8 * X[:, 0])); C = rng.exponential(2, 400)
    >>> sf = sp.survival_forest(time=np.minimum(T, C), event=(T <= C).astype(int),
    ...                         covariates=X, n_estimators=100)
    >>> sf.predict(X[:2], failure_times=[0.5, 1.0]).shape
    (2, 2)

    References
    ----------
    [@ishwaran2008random], [@athey2019generalized]
    """
    ctx = "survival_forest"
    ptype = _prediction_type(prediction_type)
    opts = ForestOptions(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_samples=max_samples,
        mtry=mtry,
        honest=honest,
        honesty_fraction=honesty_fraction,
        honesty_prune_leaves=honesty_prune_leaves,
        split_alpha=split_alpha,
        ci_group_size=1,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    opts.validate(ctx)
    arrays, names, keep, n_input = resolve_inputs(
        ctx,
        data,
        {"time": time, "event": event, "covariates": covariates},
        {"clusters": clusters, "weights": weights},
    )
    Y = arrays["time"][:, 0]
    D = arrays["event"][:, 0]
    X = arrays["covariates"]
    check_survival_inputs(Y, D, ctx)
    grid = failure_grid(Y, D, failure_times)
    cl = cluster_codes(arrays.get("clusters"), ctx)
    sw = sample_weights(arrays.get("weights"), equalize_cluster_weights, ctx)
    common = opts.engine_kwargs(cl, equalize_cluster_weights, sw)
    forest = train_survival_engine(
        X,
        Y,
        D,
        grid,
        num_trees=opts.n_estimators,
        min_node_size=opts.min_samples_leaf,
        mtry=opts.mtry,
        sample_weight=sw,
        common=common,
        max_depth=opts.max_depth,
    )
    res = SurvivalForestResult()
    res._engine = forest
    res._X = X
    res.failure_times = grid
    res.prediction_type = ptype
    res.predictions = forest.predict_survival(
        X, oob=True, nelson_aalen=ptype == "Nelson-Aalen"
    )
    res.feature_names = names["covariates"]
    res.n_obs = int(Y.size)
    res.detail = {
        "n_events": int(np.sum(D > 0.5)),
        "n_dropped_missing": int(n_input - Y.size),
        "n_clusters": None if cl is None else int(cl.max()) + 1,
        "options": dict(opts.__dict__),
    }
    return res


def check_survival_inputs(Y: np.ndarray, D: np.ndarray, ctx: str) -> None:
    if np.any(Y < 0):
        raise MethodIncompatibility(
            f"{ctx}: time values must be non-negative.",
            recovery_hint="Use observed follow-up times >= 0.",
            diagnostics={"min_time": float(np.min(Y))},
        )
    if not set(np.unique(D).tolist()).issubset({0.0, 1.0}):
        raise MethodIncompatibility(
            f"{ctx}: event must be binary 0/1.",
            recovery_hint="Encode event as 1 for an observed event, 0 for censored.",
            diagnostics={"event_values": np.unique(D)[:10].tolist()},
        )
    if not np.any(D > 0.5):
        raise DataInsufficient(
            f"{ctx}: no observed events.",
            recovery_hint="A survival forest needs at least one event.",
        )


__all__ = ["survival_forest", "SurvivalForestResult"]
