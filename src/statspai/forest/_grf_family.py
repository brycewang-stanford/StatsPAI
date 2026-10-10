"""Shared plumbing for the GRF-family forests built on :mod:`._grf_engine`.

Every forest in the family (``iv_forest``, ``multi_arm_forest``,
``lm_forest``, ``regression_forest``, ``multi_regression_forest``,
``probability_forest``, ``quantile_forest``, ``survival_forest``,
``causal_survival_forest``) shares

* input resolution: a DataFrame plus column names, or arrays;
* the engine options (the ``grf`` tuning parameters under StatsPAI names,
  see :class:`ForestOptions`);
* out-of-bag nuisance regressions fitted the way ``grf`` documents them
  (``num.trees = max(50, num.trees / 4)``, ``ci.group.size = 1``, the same
  sampling options, clusters and seed);
* observation weights, cluster-robust averages of doubly-robust scores
  and their best linear projection;
* the ``variable_importance`` of ``grf``.

The inference operators themselves live in :mod:`._grf_inference`
(``cluster_robust_vcov``, the clustered variance of a weighted mean); this
module only adapts them to score vectors so every family member reports
averages the same way ``sp.causal_forest`` does.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility
from . import _grf_engine as engine
from ._grf_inference import _clustered_mean_var, _coef_table, cluster_robust_vcov

# --------------------------------------------------------------------------- #
#  Options
# --------------------------------------------------------------------------- #


@dataclass
class ForestOptions:
    """Engine options shared by the family (grf argument in brackets).

    ``n_estimators`` [num.trees], ``min_samples_leaf`` [min.node.size],
    ``max_samples`` [sample.fraction], ``mtry`` [mtry], ``honest``
    [honesty], ``honesty_fraction``, ``honesty_prune_leaves``,
    ``split_alpha`` [alpha], ``imbalance_penalty``, ``stabilize_splits``,
    ``ci_group_size``, ``max_depth`` (no grf analogue), ``random_state``
    [seed], ``n_jobs`` [num.threads].
    """

    n_estimators: int = 2000
    min_samples_leaf: int = 5
    max_samples: float = 0.5
    mtry: Optional[int] = None
    honest: bool = True
    honesty_fraction: float = 0.5
    honesty_prune_leaves: bool = True
    split_alpha: float = 0.05
    imbalance_penalty: float = 0.0
    stabilize_splits: bool = True
    ci_group_size: int = 2
    max_depth: Optional[int] = None
    random_state: Optional[int] = 42
    n_jobs: int = 1

    def validate(self, context: str) -> None:
        def bad(name: str, value: Any, hint: str) -> MethodIncompatibility:
            return MethodIncompatibility(
                f"{context}: invalid {name}={value!r}.",
                recovery_hint=hint,
                diagnostics={name: value},
            )

        if int(self.n_estimators) < 1:
            raise bad("n_estimators", self.n_estimators, "Use n_estimators >= 1.")
        if int(self.min_samples_leaf) < 1:
            raise bad(
                "min_samples_leaf", self.min_samples_leaf, "Use min_samples_leaf >= 1."
            )
        if not 0.0 < float(self.max_samples) <= 1.0:
            raise bad("max_samples", self.max_samples, "Use 0 < max_samples <= 1.")
        if not 0.0 < float(self.honesty_fraction) < 1.0:
            raise bad(
                "honesty_fraction",
                self.honesty_fraction,
                "Use 0 < honesty_fraction < 1.",
            )
        if not 0.0 <= float(self.split_alpha) < 0.25:
            raise bad("split_alpha", self.split_alpha, "Use 0 <= split_alpha < 0.25.")
        if float(self.imbalance_penalty) < 0:
            raise bad(
                "imbalance_penalty",
                self.imbalance_penalty,
                "Use imbalance_penalty >= 0.",
            )
        if int(self.ci_group_size) < 1:
            raise bad("ci_group_size", self.ci_group_size, "Use ci_group_size >= 1.")
        if int(self.ci_group_size) > 1 and float(self.max_samples) > 0.5:
            raise bad(
                "max_samples",
                self.max_samples,
                "Confidence intervals (ci_group_size > 1) need max_samples <= 0.5.",
            )

    @property
    def seed(self) -> int:
        return 0 if self.random_state is None else int(self.random_state)

    def engine_kwargs(
        self,
        clusters: Optional[np.ndarray],
        equalize_cluster_weights: bool,
        sample_weight: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Keyword arguments shared by every engine call of one fit."""
        return dict(
            clusters=clusters,
            equalize_cluster_weights=bool(equalize_cluster_weights),
            sample_weight=sample_weight,
            sample_fraction=float(self.max_samples),
            honesty=bool(self.honest),
            honesty_fraction=float(self.honesty_fraction),
            honesty_prune_leaves=bool(self.honesty_prune_leaves),
            alpha=float(self.split_alpha),
            imbalance_penalty=float(self.imbalance_penalty),
            seed=self.seed,
            n_jobs=int(self.n_jobs),
        )


# --------------------------------------------------------------------------- #
#  Input resolution
# --------------------------------------------------------------------------- #


_ONE_HOT_NAME = re.compile(r"^(?P<col>.+)\[(?P<level>.*)\]$")


def one_hot_covariates(
    data: pd.DataFrame, cols: List[str], context: str
) -> Tuple[pd.DataFrame, List[str]]:
    """Covariate block with categorical columns as indicator columns.

    A text or ``category`` column, or an entry written ``C(col)``, becomes
    one 0/1 column per level, named ``col[level]``. Every level gets a
    column: a tree needs no reference category, and keeping all of them lets
    prediction tell an unseen level from the omitted one. Numeric columns
    pass through, and a frame with none to expand is returned as it is.
    """
    from ..core._covariates import _factor_column, _is_categorical, _levels

    out: Dict[str, Any] = {}
    names: List[str] = []
    expanded = False
    for entry in cols:
        named = _factor_column(entry)
        col = named if named is not None else entry
        if col not in data.columns:
            raise MethodIncompatibility(
                f"{context}: Missing columns: {[col]}",
                recovery_hint="Pass column names present in the input DataFrame.",
                diagnostics={"missing_columns": [col]},
            )
        series = data[col]
        if named is None and not _is_categorical(series):
            out[col] = series
            names.append(col)
            continue
        expanded = True
        missing = series.isna()
        for level in _levels(series):
            name = f"{col}[{level}]"
            out[name] = (series == level).astype(float).where(~missing)
            names.append(name)
    if not expanded:
        return data[cols], list(cols)
    return pd.DataFrame(out, index=data.index), names


def one_hot_newdata(newdata: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Rebuild the indicator columns of :func:`one_hot_covariates` on new rows.

    A feature ``col[level]`` that is not a column of ``newdata`` is computed
    from ``newdata[col]``. A value of ``col`` that matches none of the
    fitted levels is an error.
    """
    wanted: Dict[str, List[Tuple[str, str]]] = {}
    for name in feature_names:
        if name in newdata.columns:
            continue
        m = _ONE_HOT_NAME.match(name)
        if m is not None and m.group("col") in newdata.columns:
            wanted.setdefault(m.group("col"), []).append((name, m.group("level")))
    if not wanted:
        return newdata
    out = newdata.copy()
    for col, pairs in wanted.items():
        text = out[col].astype(object).map(lambda v: None if pd.isna(v) else str(v))
        known = {level for _, level in pairs}
        unseen = sorted({v for v in text.dropna().unique() if v not in known})
        if unseen:
            raise MethodIncompatibility(
                f"newdata column {col!r} has levels that were not in the "
                f"fitted data: {unseen[:5]}.",
                diagnostics={"column": col},
            )
        for name, level in pairs:
            out[name] = (text == level).astype(float).where(text.notna())
    return out


def _column_block(
    data: Optional[pd.DataFrame], spec: Any, name: str, context: str
) -> Tuple[np.ndarray, List[str]]:
    """Resolve ``spec`` (column name(s) of ``data``, or an array) to a 2-D
    float block and its column names."""
    if spec is None:
        raise MethodIncompatibility(
            f"{context}: {name}= is required.",
            recovery_hint=f"Pass {name}= as column name(s) or an array.",
        )
    if data is not None and (
        isinstance(spec, str)
        or (
            isinstance(spec, (list, tuple))
            and len(spec) > 0
            and all(isinstance(s, str) for s in spec)
        )
    ):
        cols = [spec] if isinstance(spec, str) else list(spec)
        if name == "covariates":
            # Covariates may be categorical (a text / category column, or an
            # entry written C(col)); outcomes and treatments may not.
            block, cols = one_hot_covariates(data, cols, context)
        else:
            missing = [c for c in cols if c not in data.columns]
            if missing:
                raise MethodIncompatibility(
                    f"{context}: Missing columns: {missing}",
                    recovery_hint=("Pass column names present in the input DataFrame."),
                    diagnostics={"missing_columns": missing},
                )
            block = data[cols]
        try:
            arr = block.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                f"{context}: {name} columns must be numeric.",
                recovery_hint="Encode categorical columns numerically first.",
                diagnostics={"columns": cols},
            ) from exc
        return arr, cols
    if isinstance(spec, (list, tuple)) and len(spec) == 0:
        raise MethodIncompatibility(
            f"{context}: {name}= must not be empty.",
            recovery_hint=f"Pass at least one {name} column.",
        )
    if isinstance(spec, pd.DataFrame):
        return spec.to_numpy(dtype=float), [str(c) for c in spec.columns]
    if isinstance(spec, pd.Series):
        return spec.to_numpy(dtype=float)[:, None], [str(spec.name or name)]
    arr = np.asarray(spec, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise MethodIncompatibility(
            f"{context}: {name} must be a vector or a matrix.",
            recovery_hint=f"Pass {name} as a 1-D or 2-D array.",
        )
    if arr.shape[1] == 1:
        return arr, [name]
    return arr, [f"{name}{j + 1}" for j in range(arr.shape[1])]


def resolve_inputs(
    context: str,
    data: Optional[pd.DataFrame],
    blocks: Dict[str, Any],
    optional: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]], np.ndarray, int]:
    """Resolve named blocks, drop rows with a missing value in any of them.

    Returns ``(arrays, names, kept_mask, n_input_rows)``.  ``optional``
    blocks (clusters, weights) may be ``None``; clusters may be
    non-numeric.
    """
    if data is not None and not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            f"{context}: data must be a pandas DataFrame.",
            recovery_hint="Pass a DataFrame, or data=None with arrays.",
            diagnostics={"type": type(data).__name__},
        )
    arrays: Dict[str, np.ndarray] = {}
    names: Dict[str, List[str]] = {}
    for key, spec in blocks.items():
        arr, cols = _column_block(data, spec, key, context)
        arrays[key] = arr
        names[key] = cols
    raw_optional: Dict[str, np.ndarray] = {}
    for key, spec in (optional or {}).items():
        if spec is None:
            continue
        if data is not None and isinstance(spec, str):
            if spec not in data.columns:
                raise MethodIncompatibility(
                    f"{context}: Missing columns: {[spec]}",
                    recovery_hint="Pass column names present in the input DataFrame.",
                    diagnostics={"missing_columns": [spec]},
                )
            raw_optional[key] = data[spec].to_numpy()
        else:
            raw_optional[key] = np.asarray(spec).ravel()
    n_rows = {a.shape[0] for a in arrays.values()} | {
        a.shape[0] for a in raw_optional.values()
    }
    if len(n_rows) != 1:
        raise MethodIncompatibility(
            f"{context}: inputs have different numbers of rows.",
            recovery_hint="Pass aligned arrays (one row per observation).",
            diagnostics={"row_counts": sorted(int(r) for r in n_rows)},
        )
    n = n_rows.pop()
    keep = np.ones(n, dtype=bool)
    for arr in arrays.values():
        keep &= np.isfinite(arr).all(axis=1)
    for key, arr in raw_optional.items():
        keep &= ~pd.isna(pd.Series(arr)).to_numpy()
    for key in list(arrays):
        arrays[key] = np.ascontiguousarray(arrays[key][keep])
    for key, arr in raw_optional.items():
        arrays[key] = arr[keep]
    if int(keep.sum()) < 3:
        raise DataInsufficient(
            f"{context}: fewer than 3 complete rows.",
            recovery_hint="Provide more complete rows (missing values are dropped).",
            diagnostics={"n_complete": int(keep.sum()), "n_input": int(n)},
        )
    return arrays, names, keep, int(n)


def cluster_codes(clusters: Optional[np.ndarray], context: str) -> Optional[np.ndarray]:
    if clusters is None:
        return None
    _, codes = np.unique(np.asarray(clusters), return_inverse=True)
    codes = codes.astype(np.int64).ravel()
    if int(codes.max()) + 1 < 2:
        raise DataInsufficient(
            f"{context}: clustered forests need at least two clusters.",
            recovery_hint="Pass clusters with more than one distinct id.",
        )
    return np.asarray(codes, dtype=np.int64)


def sample_weights(
    weights: Optional[np.ndarray], equalize: bool, context: str
) -> Optional[np.ndarray]:
    if weights is None:
        return None
    w: np.ndarray = np.asarray(weights, dtype=float).ravel()
    if not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise MethodIncompatibility(
            f"{context}: weights must be finite, non-negative and not all zero.",
            recovery_hint="Pass non-negative sample weights.",
        )
    if equalize:
        raise MethodIncompatibility(
            f"{context}: weights= cannot be combined with "
            "equalize_cluster_weights=True.",
            recovery_hint="Use one weighting scheme (grf imposes the same rule).",
        )
    return w


def observation_weights(
    clusters: Optional[np.ndarray],
    equalize: bool,
    weights: Optional[np.ndarray],
    n: int,
) -> np.ndarray:
    """Weights of the averages (grf ``observation_weights``), summing to 1:
    sample weights, or ``1 / cluster size`` with equalized clusters."""
    if weights is not None:
        raw = np.asarray(weights, dtype=float)
    elif clusters is not None and equalize:
        raw = 1.0 / np.bincount(clusters)[clusters]
    else:
        raw = np.ones(n)
    return np.asarray(raw / raw.sum(), dtype=float)


# --------------------------------------------------------------------------- #
#  Nuisances
# --------------------------------------------------------------------------- #


# Random-number streams of the auxiliary forests.  The engine draws the
# same subsamples for two forests grown from the same seed, so nuisance
# forests sharing the main forest's seed would have errors correlated with
# each other through identical bags; for an instrumental forest that biases
# the local covariance ratio (CATE RMSE 4.6% above grf's on the T3 design,
# gone with independent streams).  Each auxiliary forest therefore gets its
# own stream derived from the user's seed.
_STREAMS = {
    "Y_hat": 1,
    "W_hat": 2,
    "Z_hat": 3,
    "compliance": 4,
    "var_z": 5,
    "event": 6,
    "censoring": 7,
    "var_w": 8,
}


def with_stream(common: Dict[str, Any], stream: str) -> Dict[str, Any]:
    """Copy of the engine kwargs with the seed of auxiliary ``stream``."""
    out = dict(common)
    out["seed"] = (int(common.get("seed", 0)) + 7919 * _STREAMS[stream]) % (2**31 - 1)
    return out


def nuisance_trees(n_estimators: int) -> int:
    return max(50, int(n_estimators) // 4)


def oob_regression(
    X: np.ndarray,
    y: np.ndarray,
    opts: ForestOptions,
    common: Dict[str, Any],
    name: str,
    context: str,
) -> np.ndarray:
    """Out-of-bag regression-forest predictions of ``y`` on ``X``."""
    mtry = None if opts.mtry is None else int(min(max(int(opts.mtry), 1), X.shape[1]))
    forest = engine.train_forest(
        X,
        np.asarray(y, dtype=float),
        kind=engine.KIND_REGRESSION,
        num_trees=nuisance_trees(opts.n_estimators),
        mtry=mtry,
        min_node_size=5,
        ci_group_size=1,
        **with_stream(common, name),
    )
    pred, _ = forest.predict_oob(X)
    require_finite(pred, name, context)
    return np.asarray(pred, dtype=float)


def require_finite(values: np.ndarray, name: str, context: str) -> None:
    missing = int(np.sum(~np.isfinite(values)))
    if missing:
        raise DataInsufficient(
            f"{context}: the {name} forest left {missing} row(s) without an "
            "out-of-bag prediction.",
            recovery_hint=(
                "Increase n_estimators so every row is out-of-bag for some "
                "tree, or pass the nuisance estimates yourself."
            ),
            diagnostics={"n_missing": missing, "nuisance": name},
        )


def user_nuisance(
    values: Any, n: int, width: int, name: str, context: str
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0 or arr.size == 1:
        arr = np.full((n, width), float(arr.ravel()[0]))
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape != (n, width):
        raise MethodIncompatibility(
            f"{context}: {name} must have shape ({n}, {width}) (after "
            "dropping rows with missing values).",
            recovery_hint=f"Pass one {name} value per row and column.",
            diagnostics={"shape": list(arr.shape)},
        )
    if not np.isfinite(arr).all():
        raise MethodIncompatibility(
            f"{context}: {name} contains non-finite values.",
            recovery_hint=f"Pass finite {name} estimates.",
        )
    return arr


# --------------------------------------------------------------------------- #
#  Score inference
# --------------------------------------------------------------------------- #


def score_average(
    scores: np.ndarray,
    w: np.ndarray,
    clusters: Optional[np.ndarray],
    alpha: float,
) -> Dict[str, Any]:
    """Weighted mean of doubly-robust scores with grf's (cluster-robust)
    standard error: ``sum_g (sum_{i in g} w_i (s_i - mean))^2 /
    (sum w)^2 * G / (G - 1)``."""
    est = float(np.sum(w * scores) / np.sum(w))
    se = float(np.sqrt(_clustered_mean_var(scores - est, w, clusters)))
    z = float(stats.norm.ppf(1 - alpha / 2))
    zstat = est / se if se > 0 else np.nan
    return {
        "estimate": est,
        "se": se,
        "ci_low": est - z * se,
        "ci_high": est + z * se,
        "pvalue": float(2 * stats.norm.sf(abs(zstat))) if se > 0 else np.nan,
    }


def score_blp(
    scores: np.ndarray,
    A: Optional[np.ndarray],
    names: List[str],
    w: np.ndarray,
    clusters: Optional[np.ndarray],
    vcov_type: str,
    alpha: float,
    method: str,
) -> pd.DataFrame:
    """Best linear projection: weighted OLS of scores on ``(1, A)`` with a
    ``sandwich::vcovCL``-type covariance (``grf::best_linear_projection``)."""
    n = scores.size
    design = np.ones((n, 1)) if A is None else np.column_stack([np.ones(n), A])
    XtWX = design.T @ (design * w[:, None])
    beta = np.linalg.lstsq(XtWX, design.T @ (w * scores), rcond=None)[0]
    resid = scores - design @ beta
    V = cluster_robust_vcov(
        design, resid, weights=w, clusters=clusters, vcov_type=vcov_type
    )
    out = _coef_table(beta, V, names, alpha)
    out = out.drop(columns=["null"]).rename(
        columns={"ci_low": "ci_lower", "ci_high": "ci_upper"}
    )
    out.attrs["method"] = method
    return out


def validate_alpha(alpha: float, context: str) -> float:
    try:
        a = float(alpha)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: alpha must be a number in (0, 1).",
            recovery_hint="Use alpha=0.05 for 95% intervals.",
        ) from exc
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise MethodIncompatibility(
            f"{context}: alpha must be in (0, 1), got {alpha!r}.",
            recovery_hint="Use alpha=0.05 for 95% intervals.",
            diagnostics={"alpha": alpha},
        )
    return a


def validate_vcov_type(vcov_type: str, context: str) -> str:
    vt = str(vcov_type).upper()
    if vt not in ("HC0", "HC1", "HC2", "HC3"):
        raise MethodIncompatibility(
            f"{context}: unsupported vcov_type {vcov_type!r}.",
            recovery_hint="Use 'HC0', 'HC1', 'HC2' or 'HC3'.",
        )
    return vt


# --------------------------------------------------------------------------- #
#  Variable importance
# --------------------------------------------------------------------------- #


def importance_from_split_frequencies(
    counts: np.ndarray, decay_exponent: float = 2.0
) -> np.ndarray:
    """grf's variable importance from a ``(max_depth, p)`` split-count table.

    Split counts are turned into shares within each depth, the depth-``d``
    shares are weighted by ``d^(-decay_exponent)``, and the weighted sum is
    divided by the sum of the weights, so the importances sum to one when
    every depth has at least one split.  A depth without splits contributes
    zero (its weight still counts in the normaliser).
    """
    counts = np.asarray(counts, dtype=float)
    depth_totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = np.where(depth_totals > 0, counts / depth_totals, 0.0)
    weights = np.arange(1, counts.shape[0] + 1, dtype=float) ** (-float(decay_exponent))
    return np.asarray(weights @ shares / weights.sum(), dtype=float)


# --------------------------------------------------------------------------- #
#  Base class
# --------------------------------------------------------------------------- #


class GRFFamilyForest(ResultProtocolMixin):
    """Common surface of the GRF-family forests.

    Subclasses set ``_engine`` (the trained :class:`._grf_engine.GRFForest`),
    ``_X`` (training effect modifiers), ``feature_names`` and ``n_obs``.
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ("athey2019generalized",)
    _context = "forest"

    feature_names: List[str]
    n_obs: int
    _engine: Any
    _X: np.ndarray
    # Inference state set by the fitting functions (not all forests use all).
    _Y: np.ndarray
    _W: np.ndarray
    _Z: np.ndarray
    _y_hat: np.ndarray
    _w_hat: np.ndarray
    _z_hat: np.ndarray
    _z_var: np.ndarray
    _compliance: np.ndarray
    _clusters: Optional[np.ndarray]
    _obs_weight: np.ndarray
    _arm_codes: np.ndarray
    _propensity: np.ndarray
    _A: np.ndarray
    _B: np.ndarray
    _e_scores: np.ndarray
    _quantiles: np.ndarray

    def _new_X(self, newdata: Any) -> np.ndarray:
        if newdata is None:
            return self._X
        if isinstance(newdata, pd.DataFrame):
            newdata = one_hot_newdata(newdata, list(self.feature_names))
            missing = [c for c in self.feature_names if c not in newdata.columns]
            if missing:
                raise MethodIncompatibility(
                    f"{self._context}.predict(): newdata lacks columns {missing}.",
                    recovery_hint="Pass the covariate columns used at fit time.",
                )
            arr = newdata[self.feature_names].to_numpy(dtype=float)
        else:
            arr = np.asarray(newdata, dtype=float)
            if arr.ndim == 1:
                arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != len(self.feature_names):
            raise MethodIncompatibility(
                f"{self._context}.predict(): newdata must have "
                f"{len(self.feature_names)} covariate columns.",
                recovery_hint="Pass rows with the covariates used at fit time.",
                diagnostics={"shape": list(arr.shape)},
            )
        if not np.isfinite(arr).all():
            raise MethodIncompatibility(
                f"{self._context}.predict(): newdata has non-finite values.",
                recovery_hint="Drop or impute missing covariates first.",
            )
        return np.ascontiguousarray(arr)

    def _require_little_bags(self) -> None:
        """Refuse a variance request the forest cannot answer.

        Variance estimates compare little bags of trees; with
        ``ci_group_size=1`` there are none and the engine returns NaN for
        new rows, which is not an estimate.
        """
        if int(self._engine.ci_group_size) < 2:
            raise MethodIncompatibility(
                f"{self._context}: variance estimates need ci_group_size >= 2.",
                recovery_hint="Refit with ci_group_size=2.",
            )

    def split_frequencies(self, max_depth: int = 4) -> pd.DataFrame:
        """Number of splits on each covariate at each depth."""
        counts = self._engine.split_frequencies(int(max_depth))
        return pd.DataFrame(
            counts,
            columns=self.feature_names,
            index=pd.Index(range(1, counts.shape[0] + 1), name="depth"),
        )

    def variable_importance(
        self, decay_exponent: float = 2.0, max_depth: int = 4
    ) -> pd.Series:
        """``grf::variable_importance``: depth-weighted split shares."""
        counts = self._engine.split_frequencies(int(max_depth))
        imp = importance_from_split_frequencies(counts, decay_exponent)
        return pd.Series(imp, index=self.feature_names, name="importance")

    def forest_weights(self, newdata: Any = None) -> np.ndarray:
        """Dense forest weights ``alpha_i(x)`` (rows: ``newdata``, columns:
        training rows); out-of-bag when ``newdata`` is None."""
        if newdata is None:
            return np.asarray(self._engine.forest_weights(self._X, oob=True))
        return np.asarray(self._engine.forest_weights(self._new_X(newdata), oob=False))

    @property
    def num_trees(self) -> int:
        return int(self._engine.num_trees)


def as_float_vector(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def z_crit(alpha: float) -> float:
    return float(stats.norm.ppf(1 - alpha / 2))


def normal_pvalue(est: float, se: float) -> float:
    if not (se > 0 and math.isfinite(se)):
        return float("nan")
    return float(2 * stats.norm.sf(abs(est / se)))


def ensure_binary(values: np.ndarray, name: str, context: str) -> np.ndarray:
    vals = np.unique(values)
    if not np.all(np.isin(vals, (0.0, 1.0))):
        raise MethodIncompatibility(
            f"{context}: {name} must be binary 0/1 here.",
            recovery_hint=f"Encode {name} as 0/1.",
            diagnostics={f"{name}_values": vals[:10].tolist()},
        )
    return values


def names_or_default(names: Optional[Sequence[str]], p: int, stem: str) -> List[str]:
    if names is not None and len(names) == p:
        return [str(s) for s in names]
    return [f"{stem}{j + 1}" for j in range(p)]
