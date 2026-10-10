"""Prediction forests of the GRF family.

* :func:`regression_forest` -- ``E[Y | X]`` with little-bag variances
  (``grf::regression_forest``);
* :func:`multi_regression_forest` -- ``E[Y_1..Y_q | X]`` with splits that
  target all outcomes jointly (``grf::multi_regression_forest``);
* :func:`probability_forest` -- class probabilities ``P[Y = k | X]``
  (``grf::probability_forest``);
* :func:`quantile_forest` -- conditional quantiles, trees split on the
  class of ``Y`` relative to the node's quantiles and predictions are
  forest-weighted quantiles (``grf::quantile_forest``; ATW 2019, Sec. 3.3;
  Meinshausen 2006).

They are honest generalized random forests on the StatsPAI engine; the
first three also serve as the nuisance learners of the causal forests.

References
----------
[@athey2019generalized], [@meinshausen2006quantile]
"""

from __future__ import annotations

import inspect
import re
import textwrap
from typing import Any, Optional, Sequence

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


class PredictionForest(GRFFamilyForest):
    """A fitted regression / multi-regression / probability / quantile forest.

    Attributes
    ----------
    predictions : np.ndarray
        Out-of-bag predictions for the training rows: ``(n,)`` for a
        regression forest, ``(n, q)`` for multi-regression, ``(n, K)``
        class probabilities, ``(n, len(quantiles))`` quantiles.
    variance : np.ndarray or None
        Out-of-bag little-bag variances (regression, multi-regression and
        probability forests with ``ci_group_size >= 2``).
    outcome_names : list of str
        Outcome columns, class labels or quantile labels.

    Examples
    --------
    >>> import numpy as np, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 2)); y = X[:, 0] + rng.normal(size=300)
    >>> sp.regression_forest(y=y, covariates=X, n_estimators=100).forest_type
    'regression_forest'
    """

    def __init__(self, kind_name: str) -> None:
        self.forest_type = kind_name
        self._context = kind_name
        self.predictions: np.ndarray = np.zeros(0)
        self.variance: Optional[np.ndarray] = None
        self.outcome_names: list = []
        self.classes: list = []
        self.quantiles: list = []
        self.feature_names: list = []
        self.n_obs = 0
        self.detail: dict = {}

    def predict(
        self,
        newdata: Any = None,
        estimate_variance: bool = False,
        quantiles: Optional[Sequence[float]] = None,
    ) -> pd.DataFrame:
        """Predictions at ``newdata`` (out-of-bag for the training rows
        when ``newdata`` is None)."""
        if self.forest_type == "quantile_forest":
            q = self._quantiles if quantiles is None else _check_quantiles(quantiles)
            if newdata is None and quantiles is None:
                vals = self.predictions
            elif newdata is None:
                vals = self._engine.predict_quantiles(self._X, q, oob=True)
            else:
                vals = self._engine.predict_quantiles(self._new_X(newdata), q)
            return pd.DataFrame(vals, columns=[f"q{v:g}" for v in q])
        if quantiles is not None:
            raise MethodIncompatibility(
                f"{self._context}.predict(): quantiles= is for quantile forests.",
                recovery_hint="Drop quantiles=.",
            )
        if newdata is None:
            pred, var = self.predictions, self.variance
        else:
            if estimate_variance:
                self._require_little_bags()
            pred, var = self._engine.predict(
                self._new_X(newdata), estimate_variance=bool(estimate_variance)
            )
            if self.forest_type == "regression_forest":
                pred = np.asarray(pred).ravel()
                var = np.asarray(var).ravel()
        pred2 = np.asarray(pred, dtype=float)
        if pred2.ndim == 1:
            out = pd.DataFrame({"predictions": pred2})
        else:
            out = pd.DataFrame(pred2, columns=list(self.outcome_names))
        if estimate_variance:
            if var is None:
                raise MethodIncompatibility(
                    f"{self._context}: variance estimates need ci_group_size >= 2.",
                    recovery_hint="Refit with ci_group_size=2.",
                )
            v = np.asarray(var, dtype=float)
            if v.ndim == 1:
                out["variance_estimates"] = v
            else:
                for j, nm in enumerate(self.outcome_names):
                    out[f"variance_{nm}"] = v[:, j]
        return out

    def summary(self) -> str:
        return (
            f"{self.forest_type} (GRF engine)\n"
            f"  N      : {self.n_obs}\n"
            f"  trees  : {self.num_trees}\n"
            f"  outputs: {', '.join(map(str, self.outcome_names))}"
        )

    def __repr__(self) -> str:
        return f"PredictionForest({self.forest_type}, n={self.n_obs})"


def _options(**kw: Any) -> ForestOptions:
    opts = ForestOptions(**kw)
    return opts


def _check_quantiles(quantiles: Sequence[float]) -> np.ndarray:
    q = np.asarray(quantiles, dtype=float).ravel()
    if q.size == 0 or not np.all((q > 0) & (q < 1)):
        raise MethodIncompatibility(
            "quantile_forest: quantiles must lie strictly between 0 and 1.",
            recovery_hint="Use e.g. quantiles=(0.1, 0.5, 0.9).",
            diagnostics={"quantiles": q.tolist()},
        )
    return q


def _setup(  # type: ignore[no-untyped-def]
    context, data, y, covariates, clusters, weights, equalize, opts
):
    opts.validate(context)
    arrays, names, keep, n_input = resolve_inputs(
        context,
        data,
        {"y": y, "covariates": covariates},
        {"clusters": clusters, "weights": weights},
    )
    cl = cluster_codes(arrays.get("clusters"), context)
    sw = sample_weights(arrays.get("weights"), equalize, context)
    common = opts.engine_kwargs(cl, equalize, sw)
    return arrays, names, keep, n_input, cl, sw, common


def _finish(  # type: ignore[no-untyped-def]
    res, forest, X, names, n_input, opts, cl
) -> PredictionForest:
    res._engine = forest
    res._X = X
    res.feature_names = names["covariates"]
    res.n_obs = int(X.shape[0])
    res.detail = {
        "n_dropped_missing": int(n_input - X.shape[0]),
        "n_clusters": None if cl is None else int(cl.max()) + 1,
        "options": dict(opts.__dict__),
    }
    out: PredictionForest = res
    return out


_COMMON_DOC = """
    data : pd.DataFrame, optional
        Input data; when omitted, ``y`` and ``covariates`` are arrays.
    covariates : str, list of str or 2-D array
        Covariates ``X``.
    clusters : str or array-like, optional
        Cluster ids (trees sample whole clusters).
    weights : str or array-like, optional
        Sample weights (grf ``sample.weights``).
    equalize_cluster_weights : bool, default False
    n_estimators : int, default 2000
        Trees (grf ``num.trees``).
    min_samples_leaf : int, default 5
        grf ``min.node.size``.
    max_samples : float, default 0.5
        grf ``sample.fraction``.
    mtry : int, optional
    honest, honesty_fraction, honesty_prune_leaves
        grf honesty options.
    split_alpha : float, default 0.05
        grf ``alpha`` (minimum child share).
    imbalance_penalty : float, default 0
    ci_group_size : int, default 2
        Trees per little bag (variance estimates need >= 2).
    max_depth : int, optional
    random_state : int, default 42
    n_jobs : int, default 1
"""


_SHARED_OPTIONS_PLACEHOLDER = re.compile(
    r"^[ \t]*\(see below for the shared forest options[^)]*\)\n", re.M
)


def _with_shared_options(doc: Optional[str]) -> str:
    """Splice the shared forest options into a forest's docstring.

    The docstring is cleaned first. Python 3.13 strips a docstring's
    indentation at compile time and earlier versions do not, so matching the
    placeholder with its indentation found it on 3.12 and missed it on 3.13,
    where the four forests then documented ``y`` and nothing else.
    """
    cleaned = inspect.cleandoc(doc or "")
    shared = textwrap.dedent(_COMMON_DOC).lstrip("\n")
    return _SHARED_OPTIONS_PLACEHOLDER.sub(lambda _: shared, cleaned, count=1)


@accepts_aliases(_strict=True, n_trees="n_estimators")
def regression_forest(
    data: Optional[pd.DataFrame] = None,
    y: Any = None,
    covariates: Any = None,
    *,
    clusters: Any = None,
    weights: Any = None,
    equalize_cluster_weights: bool = False,
    n_estimators: int = 2000,
    min_samples_leaf: int = 5,
    max_samples: float = 0.5,
    mtry: Optional[int] = None,
    honest: bool = True,
    honesty_fraction: float = 0.5,
    honesty_prune_leaves: bool = True,
    split_alpha: float = 0.05,
    imbalance_penalty: float = 0.0,
    ci_group_size: int = 2,
    max_depth: Optional[int] = None,
    random_state: Optional[int] = 42,
    n_jobs: int = 1,
) -> PredictionForest:
    """
    Honest regression forest for ``E[Y | X]`` (``grf::regression_forest``).

    Parameters
    ----------
    y : str or array-like
        Outcome.
        (see below for the shared forest options)

    Returns
    -------
    PredictionForest
        ``.predictions`` (out-of-bag), ``.variance`` (little bags),
        ``.predict(newdata, estimate_variance=True)``,
        ``.variable_importance()``.

    Examples
    --------
    >>> import numpy as np, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(500, 3)); y = X[:, 0] ** 2 + rng.normal(size=500)
    >>> rf = sp.regression_forest(y=y, covariates=X, n_estimators=200)
    >>> rf.predict(X[:5]).shape
    (5, 1)

    References
    ----------
    [@athey2019generalized], [@wager2018estimation]
    """
    ctx = "regression_forest"
    opts = _options(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_samples=max_samples,
        mtry=mtry,
        honest=honest,
        honesty_fraction=honesty_fraction,
        honesty_prune_leaves=honesty_prune_leaves,
        split_alpha=split_alpha,
        imbalance_penalty=imbalance_penalty,
        ci_group_size=ci_group_size,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    arrays, names, keep, n_input, cl, sw, common = _setup(
        ctx, data, y, covariates, clusters, weights, equalize_cluster_weights, opts
    )
    Y = arrays["y"]
    if Y.shape[1] != 1:
        raise MethodIncompatibility(
            f"{ctx}: y must be one column.",
            recovery_hint="Use sp.multi_regression_forest for several outcomes.",
        )
    X = arrays["covariates"]
    forest = engine.train_forest(
        X,
        Y[:, 0],
        kind=engine.KIND_REGRESSION,
        num_trees=opts.n_estimators,
        mtry=opts.mtry,
        min_node_size=int(opts.min_samples_leaf),
        ci_group_size=int(opts.ci_group_size),
        max_depth=opts.max_depth,
        **common,
    )
    pred, var = forest.predict_oob(X, estimate_variance=opts.ci_group_size > 1)
    res = PredictionForest("regression_forest")
    res.predictions = np.asarray(pred, dtype=float)
    res.variance = np.asarray(var, dtype=float) if opts.ci_group_size > 1 else None
    res.outcome_names = names["y"]
    return _finish(res, forest, X, names, n_input, opts, cl)


regression_forest.__doc__ = _with_shared_options(regression_forest.__doc__)


def _multi_reg(ctx, X, Ymat, opts, common):  # type: ignore[no-untyped-def]
    return engine.train_forest(
        X,
        np.zeros(X.shape[0]),
        kind=engine.KIND_MULTI_REG,
        num_trees=opts.n_estimators,
        mtry=opts.mtry,
        min_node_size=int(opts.min_samples_leaf),
        ci_group_size=int(opts.ci_group_size),
        max_depth=opts.max_depth,
        M=Ymat,
        params=np.array([float(Ymat.shape[1])]),
        **common,
    )


@accepts_aliases(_strict=True, n_trees="n_estimators")
def multi_regression_forest(
    data: Optional[pd.DataFrame] = None,
    y: Any = None,
    covariates: Any = None,
    *,
    clusters: Any = None,
    weights: Any = None,
    equalize_cluster_weights: bool = False,
    n_estimators: int = 2000,
    min_samples_leaf: int = 5,
    max_samples: float = 0.5,
    mtry: Optional[int] = None,
    honest: bool = True,
    honesty_fraction: float = 0.5,
    honesty_prune_leaves: bool = True,
    split_alpha: float = 0.05,
    imbalance_penalty: float = 0.0,
    ci_group_size: int = 1,
    max_depth: Optional[int] = None,
    random_state: Optional[int] = 42,
    n_jobs: int = 1,
) -> PredictionForest:
    """
    Multi-task regression forest for ``E[Y_1, ..., Y_q | X]``
    (``grf::multi_regression_forest``).

    Splits maximise the summed between-child variation of all outcomes, so
    one tree serves every outcome; outcomes should be on comparable scales.

    Parameters
    ----------
    y : list of str or 2-D array
        Outcomes.
        (see below for the shared forest options; ``ci_group_size`` defaults
        to 1 as grf does not report variances here)

    Returns
    -------
    PredictionForest
        ``.predictions`` of shape ``(n, q)``.

    Examples
    --------
    >>> import numpy as np, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(500, 2))
    >>> Y = np.c_[X[:, 0], -X[:, 0]] + rng.normal(size=(500, 2))
    >>> sp.multi_regression_forest(y=Y, covariates=X,
    ...                            n_estimators=100).predictions.shape
    (500, 2)

    References
    ----------
    [@athey2019generalized]
    """
    ctx = "multi_regression_forest"
    opts = _options(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_samples=max_samples,
        mtry=mtry,
        honest=honest,
        honesty_fraction=honesty_fraction,
        honesty_prune_leaves=honesty_prune_leaves,
        split_alpha=split_alpha,
        imbalance_penalty=imbalance_penalty,
        ci_group_size=ci_group_size,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    arrays, names, keep, n_input, cl, sw, common = _setup(
        ctx, data, y, covariates, clusters, weights, equalize_cluster_weights, opts
    )
    X = arrays["covariates"]
    Ymat = np.ascontiguousarray(arrays["y"])
    forest = _multi_reg(ctx, X, Ymat, opts, common)
    pred, var = forest.predict_oob(X, estimate_variance=opts.ci_group_size > 1)
    res = PredictionForest("multi_regression_forest")
    res.predictions = np.asarray(pred, dtype=float)
    res.variance = np.asarray(var, dtype=float) if opts.ci_group_size > 1 else None
    res.outcome_names = names["y"]
    return _finish(res, forest, X, names, n_input, opts, cl)


multi_regression_forest.__doc__ = _with_shared_options(multi_regression_forest.__doc__)


@accepts_aliases(_strict=True, n_trees="n_estimators")
def probability_forest(
    data: Optional[pd.DataFrame] = None,
    y: Any = None,
    covariates: Any = None,
    *,
    clusters: Any = None,
    weights: Any = None,
    equalize_cluster_weights: bool = False,
    n_estimators: int = 2000,
    min_samples_leaf: int = 5,
    max_samples: float = 0.5,
    mtry: Optional[int] = None,
    honest: bool = True,
    honesty_fraction: float = 0.5,
    honesty_prune_leaves: bool = True,
    split_alpha: float = 0.05,
    imbalance_penalty: float = 0.0,
    ci_group_size: int = 2,
    max_depth: Optional[int] = None,
    random_state: Optional[int] = 42,
    n_jobs: int = 1,
) -> PredictionForest:
    """
    Probability forest for ``P[Y = k | X]`` (``grf::probability_forest``).

    Trees split on class counts (the Gini-type criterion of a regression
    forest on the class indicators); leaves hold class frequencies of the
    honest estimation sample, so every prediction is a proper probability
    vector.

    Parameters
    ----------
    y : str or array-like
        Class labels (any hashable values; classes are sorted).
        (see below for the shared forest options)

    Returns
    -------
    PredictionForest
        ``.predictions`` ``(n, K)`` out-of-bag probabilities with columns
        named by class; ``.classes``.

    Examples
    --------
    >>> import numpy as np, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(600, 2)); cls = (X[:, 0] > 0).astype(int)
    >>> pf = sp.probability_forest(y=cls, covariates=X, n_estimators=200)
    >>> pf.predictions.shape
    (600, 2)

    References
    ----------
    [@athey2019generalized]
    """
    ctx = "probability_forest"
    opts = _options(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_samples=max_samples,
        mtry=mtry,
        honest=honest,
        honesty_fraction=honesty_fraction,
        honesty_prune_leaves=honesty_prune_leaves,
        split_alpha=split_alpha,
        imbalance_penalty=imbalance_penalty,
        ci_group_size=ci_group_size,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    opts.validate(ctx)
    labels = y
    if data is not None and isinstance(y, str):
        if y not in data.columns:
            raise MethodIncompatibility(
                f"{ctx}: Missing columns: {[y]}",
                recovery_hint="Pass column names present in the input DataFrame.",
            )
        labels = data[y].to_numpy()
    labels = np.asarray(labels).ravel()
    codes_all, classes = _encode_classes(labels, ctx)
    arrays, names, keep, n_input = resolve_inputs(
        ctx,
        data,
        {"y": codes_all.astype(float), "covariates": covariates},
        {"clusters": clusters, "weights": weights},
    )
    cl = cluster_codes(arrays.get("clusters"), ctx)
    sw = sample_weights(arrays.get("weights"), equalize_cluster_weights, ctx)
    common = opts.engine_kwargs(cl, equalize_cluster_weights, sw)
    X = arrays["covariates"]
    codes = arrays["y"][:, 0].astype(int)
    onehot = np.zeros((codes.size, len(classes)))
    onehot[np.arange(codes.size), codes] = 1.0
    forest = _multi_reg(ctx, X, onehot, opts, common)
    pred, var = forest.predict_oob(X, estimate_variance=opts.ci_group_size > 1)
    res = PredictionForest("probability_forest")
    res.predictions = np.asarray(pred, dtype=float)
    res.variance = np.asarray(var, dtype=float) if opts.ci_group_size > 1 else None
    res.classes = list(classes)
    res.outcome_names = [str(c) for c in classes]
    names = dict(names)
    return _finish(res, forest, X, names, n_input, opts, cl)


probability_forest.__doc__ = _with_shared_options(probability_forest.__doc__)


def _encode_classes(labels: np.ndarray, ctx: str):  # type: ignore[no-untyped-def]
    mask = ~pd.isna(pd.Series(labels)).to_numpy()
    classes = np.unique(labels[mask])
    if classes.size < 2:
        raise DataInsufficient(
            f"{ctx}: need at least two classes.",
            recovery_hint="Pass labels with at least two distinct values.",
        )
    codes = np.full(labels.size, np.nan)
    codes[mask] = np.searchsorted(classes, labels[mask])
    return codes, classes


@accepts_aliases(_strict=True, n_trees="n_estimators")
def quantile_forest(
    data: Optional[pd.DataFrame] = None,
    y: Any = None,
    covariates: Any = None,
    *,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    regression_splitting: bool = False,
    clusters: Any = None,
    weights: Any = None,
    equalize_cluster_weights: bool = False,
    n_estimators: int = 2000,
    min_samples_leaf: int = 5,
    max_samples: float = 0.5,
    mtry: Optional[int] = None,
    honest: bool = True,
    honesty_fraction: float = 0.5,
    honesty_prune_leaves: bool = True,
    split_alpha: float = 0.05,
    imbalance_penalty: float = 0.0,
    max_depth: Optional[int] = None,
    random_state: Optional[int] = 42,
    n_jobs: int = 1,
) -> PredictionForest:
    """
    Quantile forest for conditional quantiles of ``Y`` (``grf::quantile_forest``).

    At each node the outcome is relabelled by the interval it falls in
    between the node's empirical ``quantiles``, and the split maximises the
    between-child separation of those class counts -- so trees chase
    changes anywhere in the conditional distribution, not only the mean
    (ATW 2019, Sec. 3.3).  Predictions are quantiles of the training
    outcomes under the forest weights (Meinshausen 2006).

    Parameters
    ----------
    y : str or array-like
        Outcome.
    quantiles : sequence of float, default (0.1, 0.5, 0.9)
        Quantiles that calibrate the splits and are predicted by default.
    regression_splitting : bool, default False
        Use ordinary regression splits instead (grf
        ``regression.splitting``; the quantile regression forest of
        Meinshausen 2006).
        (see below for the shared forest options)

    Returns
    -------
    PredictionForest
        ``.predictions`` ``(n, len(quantiles))`` out-of-bag quantiles;
        ``.predict(newdata, quantiles=...)`` for other levels.

    Examples
    --------
    >>> import numpy as np, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(800, 2))
    >>> y = X[:, 0] + (1 + (X[:, 1] > 0)) * rng.normal(size=800)
    >>> qf = sp.quantile_forest(y=y, covariates=X, n_estimators=200)
    >>> qf.predict(X[:3]).columns.tolist()
    ['q0.1', 'q0.5', 'q0.9']

    References
    ----------
    [@athey2019generalized], [@meinshausen2006quantile]
    """
    ctx = "quantile_forest"
    q = _check_quantiles(quantiles)
    opts = _options(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_samples=max_samples,
        mtry=mtry,
        honest=honest,
        honesty_fraction=honesty_fraction,
        honesty_prune_leaves=honesty_prune_leaves,
        split_alpha=split_alpha,
        imbalance_penalty=imbalance_penalty,
        ci_group_size=1,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    arrays, names, keep, n_input, cl, sw, common = _setup(
        ctx, data, y, covariates, clusters, weights, equalize_cluster_weights, opts
    )
    X = arrays["covariates"]
    Y = arrays["y"]
    if Y.shape[1] != 1:
        raise MethodIncompatibility(
            f"{ctx}: y must be one column.", recovery_hint="Pass one outcome."
        )
    yv = np.ascontiguousarray(Y[:, 0])
    params = np.concatenate([[1.0 if regression_splitting else 0.0], np.sort(q)])
    weight_vec = np.ones(yv.size) if sw is None else np.asarray(sw, dtype=float)
    forest = engine.train_forest(
        X,
        np.zeros(yv.size),
        kind=engine.KIND_QUANTILE,
        num_trees=opts.n_estimators,
        mtry=opts.mtry,
        min_node_size=int(opts.min_samples_leaf),
        ci_group_size=1,
        max_depth=opts.max_depth,
        M=yv[:, None],
        params=params,
        aux={
            "y": yv,
            "y_order": np.argsort(yv, kind="mergesort").astype(np.int64),
            "sample_weight": weight_vec,
        },
        **common,
    )
    res = PredictionForest("quantile_forest")
    res._quantiles = q
    res.predictions = forest.predict_quantiles(X, q, oob=True)
    res.outcome_names = [f"q{v:g}" for v in q]
    res.quantiles = q.tolist()
    return _finish(res, forest, X, names, n_input, opts, cl)


quantile_forest.__doc__ = _with_shared_options(quantile_forest.__doc__)

__all__ = [
    "PredictionForest",
    "regression_forest",
    "multi_regression_forest",
    "probability_forest",
    "quantile_forest",
]
