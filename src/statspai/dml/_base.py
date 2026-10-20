"""
Shared infrastructure for Double/Debiased ML estimators.

Each model-specific file (``plr.py``, ``irm.py``, ``pliv.py``,
``iivm.py``) inherits from :class:`_DoubleMLBase` and supplies its own
Neyman-orthogonal score via ``_fit_one_rep``. The base class handles
validation, default learners, repeat-split aggregation, and
:class:`CausalResult` construction.
"""

import operator
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..core._learners import positive_class_proba
from ..core._validate import treatment_as_float as _treatment_as_float
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from . import _oof_retention as _retention
from ._external_predictions import (
    validate_external_partitions,
    validate_external_prediction_input,
)
from ._learners import resolve_learner
from .oof import OOFPredictions, _attach_result_oof


def _positive_int(value: Any, *, name: str, context: str) -> int:
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise MethodIncompatibility(
            f"{context}: {name} must be a positive integer"
        ) from exc
    if isinstance(value, bool) or parsed < 1:
        raise MethodIncompatibility(f"{context}: {name} must be a positive integer")
    return int(parsed)


def _open_unit_float(value: Any, *, name: str, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: {name} must be finite and in the open interval (0, 1)"
        ) from exc
    if not np.isfinite(parsed) or not (0.0 < parsed < 1.0):
        raise MethodIncompatibility(
            f"{context}: {name} must be finite and in the open interval (0, 1)"
        )
    return parsed


def _coerce_column_list(value: Any, *, name: str, context: str) -> List[str]:
    if isinstance(value, str):
        return [value]
    try:
        cols = list(value)
    except TypeError as exc:
        raise MethodIncompatibility(
            f"{context}: {name} must be a column name or a list of column names"
        ) from exc
    if not all(isinstance(col, str) for col in cols):
        raise MethodIncompatibility(
            f"{context}: {name} must contain only column-name strings"
        )
    return cols


# Models that route explicit folds through ``_make_splits``.
_FOLD_AWARE_MODELS = frozenset({"PLR", "IRM", "PLIV", "IIVM"})


#: Printed by ``summary()`` on a PLR fit that used the default learners.
#: Numbers are the Track B coverage row and its learner experiment
#: (``tests/coverage_monte_carlo/mechanisms/dml_plr_learners.py``).
_DEFAULT_PLR_LEARNER_NOTE = (
    "default gradient boosting; 95% CI covered 0.88 on the one design "
    "tested (n=500). Pass learners or compare sp.dml_model_averaging."
)


def _splits_to_fold_labels(
    fold_indices: Any, n: int, context: str
) -> Optional[np.ndarray]:
    """Fold labels from scikit-learn style ``[(train, test), ...]`` splits.

    Returns ``None`` when ``fold_indices`` is not a sequence of index pairs,
    so the caller reads it as one label per row. The test sets must be
    positional row numbers that partition ``0 .. n - 1``; each training set
    must be the complement of its test set, because that is the only split
    the cross-fitting here will run.
    """
    if not isinstance(fold_indices, (list, tuple)) or len(fold_indices) < 2:
        return None
    if not all(isinstance(f, (list, tuple)) and len(f) == 2 for f in fold_indices):
        return None
    try:
        pairs = [
            (
                np.asarray(tr, dtype=np.int64).ravel(),
                np.asarray(te, dtype=np.int64).ravel(),
            )
            for tr, te in fold_indices
        ]
    except (TypeError, ValueError):
        return None
    if len(pairs) == n and all(tr.size == 1 and te.size == 1 for tr, te in pairs):
        return None  # n rows of two labels each is not a split
    labels = np.full(n, -1, dtype=np.int64)
    for k, (train, test) in enumerate(pairs):
        if test.size == 0 or test.min() < 0 or test.max() >= n:
            raise MethodIncompatibility(
                f"{context}: fold_indices split {k} has test rows outside "
                f"0..{n - 1} (splits are positional row numbers)."
            )
        if np.any(labels[test] >= 0):
            raise MethodIncompatibility(
                f"{context}: fold_indices test sets overlap; each row must be "
                "held out exactly once."
            )
        labels[test] = k
        expected = np.setdiff1d(np.arange(n), test)
        if not np.array_equal(np.sort(train), expected):
            raise MethodIncompatibility(
                f"{context}: in fold_indices split {k} the training rows are "
                "not the complement of the test rows. Cross-fitting trains on "
                "every row outside the held-out fold.",
                recovery_hint="Pass one fold label per row, or splits from "
                "KFold(...).split(X).",
            )
    if np.any(labels < 0):
        raise MethodIncompatibility(
            f"{context}: fold_indices test sets leave "
            f"{int(np.sum(labels < 0))} row(s) never held out."
        )
    return labels


class _DoubleMLBase:
    """Abstract base: common plumbing for all DML estimators."""

    # Overridden by subclasses
    _MODEL_TAG: str = ""  # short label, used in method= string
    _ESTIMAND: str = "ATE"  # 'ATE' or 'LATE'
    _REQUIRES_INSTRUMENT: bool = False
    # Whether ``ml_m`` / ``ml_r`` model a binary target, i.e. should
    # default to a classifier and accept binary learner aliases. Naming
    # caveat: in PLR / IRM the ml_m target is D (treatment); in IIVM it
    # is Z (instrument propensity). Hence the target-shape name, not
    # ``_BINARY_TREATMENT`` — these are nuisance-target descriptors,
    # not estimand descriptors.
    _ML_M_TARGET_BINARY: bool = False
    _ML_R_TARGET_BINARY: bool = False
    # Current IV implementations use a scalar reduced-form nuisance.
    _REQUIRES_SCALAR_INSTRUMENT: bool = True
    # Subclasses opt into sample weights when their variance supports them.
    _SUPPORTS_SAMPLE_WEIGHT: bool = False
    # Score / IPW options. ``_VALID_SCORES = None`` means the model does
    # not accept a ``score=`` argument; otherwise it is the set of legal
    # score strings and ``_DEFAULT_SCORE`` the one used when ``score`` is
    # left at ``None``. ``_USES_IPW`` marks models whose orthogonal score
    # divides by a propensity (IRM, IIVM) and therefore honour
    # ``normalize_ipw`` / ``trimming_threshold`` (DoubleML semantics).
    _VALID_SCORES: Optional[set] = None
    #: Linear score elements of the last cross-fit repetition, stashed by
    #: each model for the cluster-robust recomputation (_cluster_theta_se).
    _last_rep_score: Dict[str, Any]
    _DEFAULT_SCORE: Optional[str] = None
    _USES_IPW: bool = False

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        instrument: Optional[Union[str, List[str]]] = None,
        ml_g: Optional[Any] = None,
        ml_m: Optional[Any] = None,
        ml_r: Optional[Any] = None,
        n_folds: int = 5,
        n_rep: int = 1,
        alpha: float = 0.05,
        random_state: int = 42,
        sample_weight: Optional[Any] = None,
        fold_indices: Optional[Any] = None,
        score: Optional[str] = None,
        normalize_ipw: bool = False,
        trimming_threshold: float = 1e-2,
        cluster: Optional[Union[str, Sequence[str]]] = None,
        design: Optional[Any] = None,
    ):
        context = f"dml.{self._MODEL_TAG.lower() or 'base'}"
        if not isinstance(data, pd.DataFrame):
            raise MethodIncompatibility(
                f"{context}: data must be a pandas DataFrame",
                recovery_hint=(
                    "Pass a pandas DataFrame with named outcome, treatment, "
                    "covariate, and instrument columns."
                ),
                diagnostics={"type": type(data).__name__},
            )
        self.data = data
        self.y = y
        self.treat = treat
        self.covariates = _coerce_column_list(
            covariates,
            name="covariates",
            context=context,
        )
        if instrument is None:
            self.instrument = None
        elif isinstance(instrument, str):
            self.instrument = [instrument]
        else:
            self.instrument = _coerce_column_list(
                instrument,
                name="instrument",
                context=context,
            )
        self.n_folds = _positive_int(n_folds, name="n_folds", context=context)
        self.n_rep = _positive_int(n_rep, name="n_rep", context=context)
        self.alpha = _open_unit_float(alpha, name="alpha", context=context)
        try:
            self.random_state = int(random_state)
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                f"{context}: random_state must be integer-like"
            ) from exc

        # ----- score / IPW options (DoubleML-compatible) --------------
        if score is None:
            self.score = self._DEFAULT_SCORE
        else:
            if self._VALID_SCORES is None:
                raise MethodIncompatibility(
                    f"{context}: model='{self._MODEL_TAG.lower()}' does not accept "
                    f"a 'score' argument."
                )
            score_str = str(score)
            if score_str not in self._VALID_SCORES:
                raise MethodIncompatibility(
                    f"{context}: score must be one of {sorted(self._VALID_SCORES)}, "
                    f"got {score_str!r}.",
                    diagnostics={"valid_scores": sorted(self._VALID_SCORES)},
                )
            self.score = score_str
        if not isinstance(normalize_ipw, bool):
            raise MethodIncompatibility(
                f"{context}: normalize_ipw must be a bool, got "
                f"{type(normalize_ipw).__name__}."
            )
        if normalize_ipw and not self._USES_IPW:
            raise MethodIncompatibility(
                f"{context}: normalize_ipw only applies to inverse-propensity "
                f"scores (model in {{'irm', 'iivm'}}); model="
                f"'{self._MODEL_TAG.lower()}' does not divide by a propensity."
            )
        self.normalize_ipw = normalize_ipw
        try:
            tt = float(trimming_threshold)
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                f"{context}: trimming_threshold must be a float in (0, 0.5)."
            ) from exc
        if not (0.0 < tt < 0.5):
            raise MethodIncompatibility(
                f"{context}: trimming_threshold must be in the open interval "
                f"(0, 0.5), got {tt}."
            )
        if tt != 1e-2 and not self._USES_IPW:
            raise MethodIncompatibility(
                f"{context}: trimming_threshold only applies to IPW-based models "
                f"(model in {{'irm', 'iivm'}}); model="
                f"'{self._MODEL_TAG.lower()}' does not trim a propensity."
            )
        self.trimming_threshold = tt

        # Guard against a future model class silently ignoring caller-supplied
        # folds: every tag listed here must actually route fold_indices into
        # _make_splits(). All four current models do.
        if fold_indices is not None and self._MODEL_TAG not in _FOLD_AWARE_MODELS:
            raise MethodIncompatibility(
                f"{context}: explicit fold_indices are supported for model in "
                f"{{{', '.join(sorted(m.lower() for m in _FOLD_AWARE_MODELS))}}}; "
                f"model='{self._MODEL_TAG.lower()}' would ignore them."
            )
        if fold_indices is None:
            self._fold_indices_input: Any = None
        elif isinstance(fold_indices, str):
            if fold_indices not in data.columns:
                raise MethodIncompatibility(
                    f"{context}: fold_indices column '{fold_indices}' not in data",
                    diagnostics={"missing_columns": [fold_indices]},
                )
            self._fold_indices_input = fold_indices
        elif (
            isinstance(fold_indices, (np.ndarray, pd.DataFrame))
            and np.ndim(fold_indices) == 2
            and np.shape(fold_indices)[1] == 2
        ):
            # two-way cluster=: the fold of each unit in each dimension
            arr2 = np.asarray(fold_indices)
            if len(arr2) != len(data):
                raise MethodIncompatibility(
                    f"{context}: two-way fold_indices must have {len(data)} "
                    f"rows; got shape {arr2.shape}"
                )
            self._fold_indices_input = arr2
        else:
            labels = _splits_to_fold_labels(fold_indices, len(data), context)
            arr = np.asarray(fold_indices) if labels is None else labels
            if arr.ndim != 1 or len(arr) != len(data):
                raise MethodIncompatibility(
                    f"{context}: fold_indices must be 1-D of length {len(data)} "
                    f"(matching data); got shape {arr.shape}"
                )
            self._fold_indices_input = arr
        # Resolve sample_weight: accept Series, ndarray, or column name.
        if sample_weight is None:
            self._sample_weight_input: Any = None
        elif isinstance(sample_weight, str):
            if sample_weight not in data.columns:
                raise MethodIncompatibility(
                    f"{context}: sample_weight column '{sample_weight}' not in data",
                    diagnostics={"missing_columns": [sample_weight]},
                )
            self._sample_weight_input = sample_weight
        else:
            try:
                arr = np.asarray(sample_weight, dtype=float)
            except (TypeError, ValueError) as exc:
                raise MethodIncompatibility(
                    f"{context}: sample_weight must be numeric"
                ) from exc
            if arr.ndim != 1 or len(arr) != len(data):
                raise MethodIncompatibility(
                    f"{context}: sample_weight must be 1-D of length {len(data)} "
                    f"(matching data); got shape {arr.shape}"
                )
            self._sample_weight_input = arr
        # One- and two-way clustering (Chiang, Kato, Ma and Sasaki 2022, as
        # DoubleML implements it): folds are formed over clusters (two-way:
        # the K x K product of the two cluster partitions) and the variance
        # is built from within-cluster score sums; see _cluster_theta_se.
        self._cluster_cols: List[str] = []
        if cluster is None:
            self.cluster: Optional[Any] = None
        else:
            cols = [cluster] if isinstance(cluster, str) else list(cluster)
            if not 1 <= len(cols) <= 2 or not all(isinstance(c, str) for c in cols):
                raise MethodIncompatibility(
                    f"{context}: cluster must name one column or a list of two "
                    "(two-way clustering); more than two ways is not "
                    "implemented.",
                    diagnostics={"cluster": repr(cluster)},
                )
            missing = [c for c in cols if c not in data.columns]
            if missing:
                raise MethodIncompatibility(
                    f"{context}: cluster column(s) {missing} not in data",
                    diagnostics={"missing_columns": missing},
                )
            if len(cols) == 2 and cols[0] == cols[1]:
                raise MethodIncompatibility(
                    f"{context}: the two cluster columns are the same.",
                    recovery_hint="Pass cluster='col' for one-way clustering.",
                )
            self._cluster_cols = cols
            self.cluster = cols[0] if len(cols) == 1 else cols
            if len(cols) == 2 and design is not None:
                raise MethodIncompatibility(
                    f"{context}: two-way cluster= is not combined with design=.",
                    recovery_hint="Put the clustering in the survey design.",
                )
        self._cluster_codes: Optional[np.ndarray] = None
        self._design: Any = None
        self._design_rows: Optional[np.ndarray] = None
        if design is not None:
            from ..survey.linearization import check_design

            self._design = check_design(
                design,
                data,
                function=context,
                weights=None if sample_weight is None else "sample_weight",
                cluster=cluster,
            )
            if not self._SUPPORTS_SAMPLE_WEIGHT:  # pragma: no cover
                raise MethodIncompatibility(
                    f"{context}: design= needs a weighted score, which this "
                    "model does not have."
                )
        if self._sample_weight_input is not None and not self._SUPPORTS_SAMPLE_WEIGHT:
            raise MethodIncompatibility(  # pragma: no cover
                f"{context}: sample_weight is not yet supported for "
                f"model='{self._MODEL_TAG.lower()}'. Weighted support is "
                f"currently implemented for model in "
                f"{{'plr', 'irm', 'pliv', 'iivm'}} only."
            )

        self._validate()

        # Recorded for sp.validation_scope: coverage evidence is specific
        # to the default learners, and a user-supplied learner is not them
        # even when it has the same class.
        self._default_learners = ml_g is None and ml_m is None and ml_r is None
        self.ml_g = (
            self._default_ml_g()
            if ml_g is None
            else resolve_learner(ml_g, kind="regressor", role="ml_g")
        )
        self.ml_m = (
            self._default_ml_m()
            if ml_m is None
            else resolve_learner(
                ml_m,
                kind="classifier" if self._ML_M_TARGET_BINARY else "regressor",
                role="ml_m",
            )
        )
        self.ml_r = (
            self._default_ml_r()
            if ml_r is None
            else resolve_learner(
                ml_r,
                kind="classifier" if self._ML_R_TARGET_BINARY else "regressor",
                role="ml_r",
            )
        )

    def _validate(self) -> None:
        context = f"dml.{self._MODEL_TAG.lower() or 'base'}"
        required = [self.y, self.treat] + self.covariates
        if self.instrument is not None:
            required = required + self.instrument
        missing = [col for col in required if col not in self.data.columns]
        if missing:
            raise MethodIncompatibility(
                f"{context}: columns not found in data: {missing}",
                recovery_hint=(
                    "Check y, treat, covariates, and instrument column names."
                ),
                diagnostics={"missing_columns": missing},
            )
        if self._REQUIRES_INSTRUMENT and not self.instrument:
            raise MethodIncompatibility(
                f"{context}: model='{self._MODEL_TAG.lower()}' requires an "
                f"'instrument' argument"
            )
        if not self._REQUIRES_INSTRUMENT and self.instrument is not None:
            raise MethodIncompatibility(
                f"{context}: 'instrument' is only valid when model requires an IV "
                f"(got model='{self._MODEL_TAG.lower()}')"
            )
        if (
            self._REQUIRES_INSTRUMENT
            and self._REQUIRES_SCALAR_INSTRUMENT
            and self.instrument is not None
            and len(self.instrument) > 1
        ):
            raise MethodIncompatibility(
                f"{context}: model='{self._MODEL_TAG.lower()}' accepts a single scalar "
                f"instrument; got {len(self.instrument)}: {self.instrument}. "
                f"For multiple excluded instruments, use "
                f"sp.scalar_iv_projection(data, treat=..., "
                f"instruments={self.instrument!r}, covariates=...) "
                f"to build a scalar first-stage index column, then pass "
                f"its name to the `instrument=` argument."
            )
        if self.n_folds < 2:
            raise MethodIncompatibility(
                f"{context}: n_folds must be >= 2, got {self.n_folds}"
            )

    def _default_ml_g(self) -> Any:
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )

    def _default_ml_m(self) -> Any:
        if self._ML_M_TARGET_BINARY:
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            )
        return self._default_ml_g()

    def _default_ml_r(self) -> Any:
        if self._ML_R_TARGET_BINARY:
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            )
        return self._default_ml_g()

    # Subclasses implement this: return (theta, se) for ONE rep.
    # Subclasses may additionally populate ``self._last_rep_diagnostics``
    # (a dict) inside ``_fit_one_rep``; the base class merges those into
    # the final ``model_info['diagnostics']`` block. Default = no diags.
    # ``sample_weight`` is the dropna-aligned weight vector (same length
    # as Y/D/X). Subclasses that opt in to weighting set
    # ``_SUPPORTS_SAMPLE_WEIGHT = True`` and use ``sample_weight`` in
    # both the nuisance fits and the moment equation.
    def _fit_one_rep(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
        Z: Any,
        n: int,
        rng_seed: int,
        sample_weight: Optional[np.ndarray] = None,
        fold_indices: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        raise NotImplementedError  # pragma: no cover

    def _fit_external_rep(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
        predictions: "OOFPredictions",
        rep: int,
    ) -> Tuple[float, float]:
        del Y, D, X, predictions, rep
        raise NotImplementedError(
            "external predictions are not implemented for this DML model"
        )

    def _cluster_theta_se(self, score: Dict[str, Any]) -> Tuple[float, float]:
        """One-way cluster-robust DML estimate and SE from linear scores.

        With ``psi = psi_a * theta + psi_b`` (weighted by ``w`` when sample
        weights are used), folds ``k = 1..K`` partitioning the ``G``
        clusters into sets ``I_k`` (Chiang, Kato, Ma and Sasaki 2022; the
        one-way case of DoubleML's cluster data):

        * ``theta = -sum_k |I_k|^-1 sum_{test_k} psi_b / sum_k |I_k|^-1
          sum_{test_k} psi_a``;
        * ``J = K^-1 sum_k |I_k|^-1 sum_{test_k} psi_a``;
        * ``Gamma = K^-1 sum_k |I_k|^-1 sum_{g in I_k} (sum_{i in g} psi_i)^2``;
        * ``Var(theta) = Gamma / (J^2 G)``.
        """
        codes = self._cluster_codes
        assert codes is not None
        psi_a = np.asarray(score["psi_a"], dtype=float)
        psi_b = np.asarray(score["psi_b"], dtype=float)
        w = score.get("weights")
        if w is not None:
            w = np.asarray(w, dtype=float)
            w = w * (len(w) / w.sum())
            psi_a, psi_b = w * psi_a, w * psi_b
        if self._is_twoway():
            return self._twoway_theta_se(psi_a, psi_b)
        splits = score["splits"]
        sizes = [len(np.unique(codes[test])) for _, test in splits]
        num = sum(np.sum(psi_b[test]) / m for (_, test), m in zip(splits, sizes))
        den = sum(np.sum(psi_a[test]) / m for (_, test), m in zip(splits, sizes))
        if abs(den) < 1e-12:
            raise RuntimeError(
                "DML cluster: score denominator is ~0."
            )  # pragma: no cover
        theta = float(-num / den)
        psi = psi_a * theta + psi_b
        K = len(splits)
        J = den / K
        gamma = 0.0
        for (_, test), m in zip(splits, sizes):
            sums = np.bincount(codes[test], weights=psi[test])
            gamma += float(np.sum(sums**2)) / m
        gamma /= K
        G = int(codes.max()) + 1
        return theta, float(np.sqrt(gamma / (J**2 * G)))

    def _design_theta_se(self, score: Dict[str, Any]) -> Tuple[float, float]:
        """Design-weighted DML estimate and its survey linearisation SE.

        ``theta = -sum w psi_b / sum w psi_a`` (the weighted DML2 solution
        each model already returns) and the contributions
        ``u_i = -w_i psi_i(theta) / sum_j w_j psi_a_j`` -- the influence
        function of the weighted moment with the cross-fitted nuisances
        held fixed (the DML convention) -- go to the design variance over
        strata, PSUs and the fpc (R ``survey::svyrecvar``).
        """
        from ..survey.linearization import design_vcov

        assert self._design is not None and self._design_rows is not None
        psi_a = np.asarray(score["psi_a"], dtype=float)
        psi_b = np.asarray(score["psi_b"], dtype=float)
        w = np.asarray(score["weights"], dtype=float)
        den = float(np.sum(w * psi_a))
        if abs(den) < 1e-12 * max(float(np.sum(w * np.abs(psi_a))), 1e-300):
            raise DataInsufficient(  # pragma: no cover
                "DML design: the weighted score denominator is ~0.",
                recovery_hint="Check the treatment residual variation.",
            )
        theta = float(-np.sum(w * psi_b) / den)
        u = -w * (psi_a * theta + psi_b) / den
        V = design_vcov(u, self._design, self._design_rows)
        return theta, float(np.sqrt(V[0, 0]))

    def _is_twoway(self) -> bool:
        codes = getattr(self, "_cluster_codes", None)
        return codes is not None and np.ndim(codes) == 2

    def _twoway_splits(
        self, rng_seed: int, fold_indices: Optional[np.ndarray]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """The ``K x K`` cross-fitting cells of two-way clustering.

        Each cluster dimension is split into ``K`` folds of whole clusters;
        cell ``(k, l)`` tests the units whose first cluster is in fold
        ``k`` *and* whose second is in fold ``l``, and trains on the units
        in neither (units sharing a cluster with the test cell in either
        dimension are left out of its training set). This is DoubleML's
        multiway cluster resampling. ``fold_indices`` (``n x 2``) fixes the
        per-dimension folds.
        """
        from sklearn.model_selection import KFold

        codes = self._cluster_codes
        assert codes is not None
        K = self.n_folds
        if fold_indices is None:
            folds = np.empty_like(codes)
            for j in range(2):
                n_cl = int(codes[:, j].max()) + 1
                f_cl = np.empty(n_cl, dtype=int)
                kf = KFold(n_splits=K, shuffle=True, random_state=rng_seed + 7919 * j)
                for f, (_, test_cl) in enumerate(kf.split(np.zeros(n_cl))):
                    f_cl[test_cl] = f
                folds[:, j] = f_cl[codes[:, j]]
        else:
            folds = np.asarray(fold_indices, dtype=int)
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        cells: List[Tuple[np.ndarray, np.ndarray]] = []
        for k in range(K):
            for m in range(K):
                test = (folds[:, 0] == k) & (folds[:, 1] == m)
                train = (folds[:, 0] != k) & (folds[:, 1] != m)
                if not test.any() or not train.any():
                    raise DataInsufficient(
                        f"DML two-way cluster=: cross-fitting cell ({k}, {m}) "
                        "has no test or no training units.",
                        recovery_hint="Reduce n_folds or use one-way clustering.",
                        diagnostics={"cell": [k, m]},
                    )
                splits.append((np.flatnonzero(train), np.flatnonzero(test)))
                cells.append(
                    (
                        np.unique(codes[folds[:, 0] == k, 0]),
                        np.unique(codes[folds[:, 1] == m, 1]),
                    )
                )
        self._twoway_cells = cells
        return splits

    def _twoway_theta_se(
        self, psi_a: np.ndarray, psi_b: np.ndarray
    ) -> Tuple[float, float]:
        """Two-way cluster-robust DML estimate and SE (Chiang, Kato, Ma
        and Sasaki 2022, as DoubleML implements it).

        With cells ``(I_k, J_l)`` (sets of first- and second-dimension
        clusters, ``K`` folds per dimension) and ``c_kl = |I_k| |J_l|``:

        * ``theta = -sum_kl c_kl^-1 sum_{test} psi_b / sum_kl c_kl^-1
          sum_{test} psi_a``, ``J = K^-2 sum_kl c_kl^-1 sum_{test} psi_a``;
        * ``Gamma = K^-2 sum_kl min(|I_k|, |J_l|) / c_kl^2 [sum_{i in I_k}
          (sum_{test, c1 = i} psi)^2 + sum_{j in J_l} (sum_{test, c2 = j}
          psi)^2]``;
        * ``Var(theta) = Gamma / (J^2 min(G_1, G_2))``.
        """
        codes = self._cluster_codes
        cells = self._twoway_cells
        splits = self._last_rep_score["splits"]
        K = self.n_folds
        sizes = [len(c1) * len(c2) for c1, c2 in cells]
        num = sum(np.sum(psi_b[t]) / c for (_, t), c in zip(splits, sizes))
        den = sum(np.sum(psi_a[t]) / c for (_, t), c in zip(splits, sizes))
        if abs(den) < 1e-12:  # pragma: no cover
            raise RuntimeError("DML cluster: score denominator is ~0.")
        theta = float(-num / den)
        psi = psi_a * theta + psi_b
        gamma = 0.0
        for (_, t), (c1, c2) in zip(splits, cells):
            const = min(len(c1), len(c2)) / float(len(c1) * len(c2)) ** 2
            s1 = np.bincount(codes[t, 0], weights=psi[t])
            s2 = np.bincount(codes[t, 1], weights=psi[t])
            gamma += const * (float(np.sum(s1**2)) + float(np.sum(s2**2)))
        J_hat = den / K**2
        gamma /= K**2
        G = min(int(codes[:, 0].max()) + 1, int(codes[:, 1].max()) + 1)
        return theta, float(np.sqrt(gamma / (J_hat**2 * G)))

    def _sensitivity_rep(self, theta: float, se: float) -> Dict[str, Any]:
        """Elements of the omitted-variable-bias bound for one repetition.

        Returns the point estimate, its influence function ``psi`` (scaled
        so that ``Var(theta) = mean(psi^2)/n`` without clustering),
        ``sigma^2``, ``nu^2`` and their centred score elements. With
        ``cluster=`` it also carries what the cluster-robust variance of a
        linear combination of those scores needs.
        """
        sens = self._last_rep_sens
        score = self._last_rep_score
        psi_a = np.asarray(score["psi_a"], dtype=float)
        psi_b = np.asarray(score["psi_b"], dtype=float)
        psi = psi_a * theta + psi_b
        if "sigma2_el" in sens:
            sigma2_el = np.asarray(sens["sigma2_el"], dtype=float)
            nu2_el = np.asarray(sens["nu2_el"], dtype=float)
            nu2 = float(np.mean(nu2_el))
            psi_nu2 = nu2_el - nu2
        else:
            sigma2_el = (sens["a"] - theta * sens["b"]) ** 2
            v2 = np.asarray(sens["treat_resid"], dtype=float) ** 2
            nu2 = float(1.0 / np.mean(v2))
            psi_nu2 = nu2 - v2 * nu2**2
        sigma2 = float(np.mean(sigma2_el))
        out: Dict[str, Any] = {
            "theta": float(theta),
            "se": float(se),
            "sigma2": sigma2,
            "nu2": nu2,
            "psi_sigma2": sigma2_el - sigma2,
            "psi_nu2": psi_nu2,
        }
        codes = self._cluster_codes
        if codes is None:
            out["psi"] = -psi / float(np.mean(psi_a))
        else:
            splits = score["splits"]
            sizes = [len(np.unique(codes[test])) for _, test in splits]
            J = sum(np.sum(psi_a[test]) / m for (_, test), m in zip(splits, sizes))
            out["psi"] = -psi / (J / len(splits))
            # sigma^2 and nu^2 are averages over observations while the
            # cluster variance sums scores within clusters, so their score
            # elements carry the same observations-per-cluster scaling.
            per_cluster = np.mean([len(t) / m for (_, t), m in zip(splits, sizes)])
            scale = float(1.0 / per_cluster)
            out["psi_sigma2"] = out["psi_sigma2"] * scale
            out["psi_nu2"] = out["psi_nu2"] * scale
            out["cluster"] = {
                "codes": codes,
                "tests": [test for _, test in splits],
                "sizes": sizes,
            }
        return out

    def _make_splits(
        self,
        X: np.ndarray,
        *,
        rng_seed: int,
        fold_indices: Optional[np.ndarray] = None,
        stratify: Optional[np.ndarray] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Cross-fitting splits, honouring caller-supplied folds.

        With ``fold_indices`` the split is fully determined by the data,
        which is what makes a bit-exact comparison against another
        implementation possible: both engines then cross-fit on the same
        partition and the only remaining difference is the estimator.
        Without it, fall back to ``KFold`` — or ``StratifiedKFold`` when
        ``stratify`` is given, as the binary-nuisance models need.

        Supplied folds are used verbatim: stratification then becomes the
        caller's responsibility, so this checks that every training set
        still contains both classes rather than letting the classifier
        fail somewhere deeper with a less legible message.
        """
        if self._is_twoway():
            return self._twoway_splits(rng_seed, fold_indices)
        if fold_indices is not None:
            splits = [
                (
                    np.flatnonzero(fold_indices != fold),
                    np.flatnonzero(fold_indices == fold),
                )
                for fold in range(self.n_folds)
            ]
            if stratify is not None:
                for fold, (train_idx, _) in enumerate(splits):
                    classes = np.unique(stratify[train_idx])
                    if len(classes) < 2:
                        raise DataInsufficient(
                            f"{self._MODEL_TAG}: the supplied fold_indices leave "
                            f"training fold {fold} with a single class in the "
                            f"binary nuisance target, so its classifier is not "
                            f"identified.",
                            recovery_hint=(
                                "Supply stratified folds (each training set must "
                                "contain both 0 and 1), or drop fold_indices to "
                                "use the built-in StratifiedKFold."
                            ),
                            diagnostics={
                                "fold": fold,
                                "classes_in_train": [float(c) for c in classes],
                            },
                        )
            return splits
        if self._cluster_codes is not None:
            # Folds over whole clusters: no cluster is split between a
            # training and a test set (DoubleML's one-way cluster resampling,
            # which does not stratify either).
            from sklearn.model_selection import KFold

            codes = self._cluster_codes
            n_cl = int(codes.max()) + 1
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=rng_seed)
            return [
                (
                    np.flatnonzero(~np.isin(codes, test_cl)),
                    np.flatnonzero(np.isin(codes, test_cl)),
                )
                for _, test_cl in kf.split(np.zeros(n_cl))
            ]
        if stratify is None:
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=rng_seed)
            return list(kf.split(X))
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=rng_seed
        )
        return list(skf.split(X, stratify))

    @staticmethod
    def _validate_fold_indices(
        fold_indices: Any,
        n: int,
        n_folds: int,
    ) -> np.ndarray:
        raw = np.asarray(fold_indices)
        if raw.ndim != 1 or len(raw) != n:
            raise MethodIncompatibility(
                f"fold_indices must be length {n} after dropping missing "
                f"model rows; got shape {raw.shape}"
            )
        codes, _ = pd.factorize(raw, sort=True, use_na_sentinel=True)
        if (codes < 0).any():
            raise MethodIncompatibility("fold_indices contain missing values")
        unique = np.unique(codes)
        if len(unique) != n_folds:
            raise MethodIncompatibility(
                f"fold_indices define {len(unique)} folds, but n_folds=" f"{n_folds}"
            )
        counts = np.bincount(codes, minlength=n_folds)
        if np.any(counts == 0):
            raise DataInsufficient("fold_indices must assign at least one row per fold")
        return np.asarray(codes, dtype=int)

    # ----- Nuisance prediction (used by subclasses) -------------------
    @staticmethod
    def _predict_nuisance(
        fitted: Any, X: np.ndarray, target: np.ndarray, role: str
    ) -> np.ndarray:
        """Conditional-mean prediction of a fitted nuisance learner.

        A regressor predicts with ``predict``. A classifier is only a
        conditional-mean estimator for a 0/1 target, and then the mean
        is ``predict_proba[:, 1]``; ``predict`` would return the hard
        label, and a residual ``D - 1{p > 0.5}`` is not ``D - E[D|X]``.
        ``doubleml`` applies the same rule to PLR.
        """
        from sklearn.base import is_classifier

        if not is_classifier(fitted):
            return np.asarray(fitted.predict(X), dtype=float).ravel()
        values = np.unique(np.asarray(target))
        if not (np.isin(values, (0, 1)).all() and hasattr(fitted, "predict_proba")):
            raise MethodIncompatibility(
                f"dml: the {role} learner {type(fitted).__name__} is a "
                "classifier, but its target is not a 0/1 variable (or the "
                "learner has no predict_proba). Pass a regressor."
            )
        # (n,) or (n, 2) by classes_; a fold with one class gives 0 or 1
        return positive_class_proba(fitted, X, context=f"dml {role}")

    # ----- Sample-weight helpers (used by subclasses) -----------------
    @staticmethod
    def _fit_weighted(
        learner: Any,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray],
    ) -> Any:
        """Fit ``learner`` on (X, y); pass ``weights`` if supported.

        sklearn estimators almost universally accept ``sample_weight``
        in ``.fit``, but a few (e.g. some custom wrappers) do not. We
        try the weighted call first and fall back to unweighted with a
        one-time warning if the learner doesn't accept the kwarg.
        """
        from sklearn.base import clone

        clf = clone(learner)
        if weights is None:
            clf.fit(X, y)
            return clf
        try:
            clf.fit(X, y, sample_weight=weights)
        except TypeError:  # pragma: no cover
            # Learner doesn't support sample_weight — fall back to
            # unweighted fit. The downstream weighted moment / variance
            # is still applied; this only loses efficiency in nuisance.
            import warnings  # pragma: no cover

            warnings.warn(  # pragma: no cover
                f"{type(learner).__name__}.fit does not accept "
                f"sample_weight; falling back to unweighted nuisance "
                f"fit. The weighted moment equation is still applied.",
                RuntimeWarning,
                stacklevel=3,
            )
            clf.fit(X, y)
        return clf

    def _warn_propensity_clipping(self, diagnostics: dict, n: int) -> None:
        """Say how many cross-fitted propensities were clipped.

        The clip changes the score on those rows, so the estimate is no
        longer the AIPW formula there; ``sp.aipw`` and ``sp.tmle`` say so
        and so does this. Counts are summed over ``n_rep`` repetitions, so
        the share divides by ``n * n_rep``.
        """
        if not self._USES_IPW:
            return
        lo = diagnostics.get("n_clipped_below")
        hi = diagnostics.get("n_clipped_above")
        if not isinstance(lo, int) or not isinstance(hi, int):
            return
        total = lo + hi
        denom = max(int(n) * int(self.n_rep), 1)
        share = total / denom
        diagnostics["n_clipped"] = int(total)
        diagnostics["clip_share"] = float(share)
        if total == 0:
            return
        import warnings

        from ..exceptions import AssumptionWarning

        thr = self.trimming_threshold
        reps = f" (summed over n_rep={self.n_rep})" if self.n_rep > 1 else ""
        warnings.warn(
            f"dml.{self._MODEL_TAG.lower()}: {total} cross-fitted propensity "
            f"scores{reps} ({100.0 * share:.1f}%) lie outside "
            f"[{thr:g}, {1.0 - thr:g}] and were clipped to it, so the score "
            "differs from the unclipped formula on those rows and overlap is "
            "weak. See result.model_info['diagnostics'] and "
            "result.violations(); consider sp.overlap_weights or restricting "
            "the sample to the overlap region.",
            AssumptionWarning,
            stacklevel=4,
        )

    @staticmethod
    def _aggregate_diagnostics(per_rep: List[dict]) -> dict:
        """Merge per-rep diagnostics into a single dict.

        Numeric scalars are averaged; integer counts are summed; lists
        of fold-level scalars are concatenated. Keys not present in
        every rep are passed through untouched (last value wins).
        """
        if not per_rep:
            return {}  # pragma: no cover
        merged: dict = {}
        keys = set().union(*(d.keys() for d in per_rep))
        for k in keys:
            vals = [d[k] for d in per_rep if k in d]
            if not vals:
                continue  # pragma: no cover
            sample = vals[0]
            if isinstance(sample, bool):
                merged[k] = any(vals)
            elif isinstance(sample, int):
                merged[k] = int(sum(vals))
            elif isinstance(sample, float):
                # NaN-safe mean across reps
                arr = np.asarray(vals, dtype=float)
                if np.all(np.isnan(arr)):
                    merged[k] = float("nan")
                else:
                    merged[k] = float(np.nanmean(arr))
            elif isinstance(sample, (list, tuple)):
                acc: list = []
                for v in vals:
                    acc.extend(list(v))
                merged[k] = acc
            else:
                merged[k] = sample
        return merged

    def fit(
        self,
        *,
        external_predictions: Optional["OOFPredictions"] = None,
        store_oof: bool = False,
        observation_ids: Optional[Sequence[str]] = None,
    ) -> CausalResult:
        """Fit the configured DML model and optionally retain IRM ATE records.

        Parameters
        ----------
        external_predictions : OOFPredictions or None, default None
            Validated in-memory predictions aligned exactly to observed rows.
            Supported for unweighted, unnormalised binary IRM ATE only.
        store_oof : bool, default False
            Retain all repeats for ``get_oof()`` and ``get_residuals()``.
            Internal retained fits reject subgroup-mean fallback.
        observation_ids : sequence of str or None, default None
            Unique IDs in input row order; generated ordinal IDs otherwise.

        Returns
        -------
        CausalResult
            Effect and uncertainty with optional independent OOF snapshot.

        Examples
        --------
        >>> import statspai as sp
        >>> callable(sp.DoubleMLIRM.fit)
        True

        References
        ----------
        See ``docs/dml_oof_audit.md`` for the versioned audit contract.
        """
        _retention.validate_oof_request_scope(
            model_tag=self._MODEL_TAG,
            score=self.score,
            normalize_ipw=self.normalize_ipw,
            has_sample_weight=self._sample_weight_input is not None,
            external_predictions=external_predictions,
            store_oof=store_oof,
            observation_ids=observation_ids,
        )
        internal_explicit = (
            external_predictions is None and self._fold_indices_input is not None
        )
        if internal_explicit and self.n_rep != 1:
            context = f"dml.{self._MODEL_TAG.lower() or 'base'}"
            raise MethodIncompatibility(
                f"{context}: explicit fold_indices require n_rep=1; pass one "
                "fold assignment for the single cross-fit repetition."
            )
        collector = None
        self._cluster_codes = None
        if external_predictions is not None and self._design is not None:
            raise MethodIncompatibility(
                "DML design= is not supported with external_predictions: the "
                "design weights enter the nuisance fits and the folds are "
                "formed over PSUs here."
            )
        if external_predictions is not None and self.cluster is not None:
            raise MethodIncompatibility(
                "DML cluster= is not supported with external_predictions: the "
                "cross-fitting folds must be formed over clusters here."
            )
        if external_predictions is not None:
            analysis = self.data[[self.y, self.treat] + self.covariates]
            fi = self._fold_indices_input
            raw_fold = (
                self.data[fi].to_numpy()
                if isinstance(fi, str)
                else np.asarray(fi) if fi is not None else None
            )
            external_input = validate_external_prediction_input(
                external_predictions,
                y_values=analysis[self.y].to_numpy(),
                d_values=analysis[self.treat].to_numpy(),
                x_columns=[
                    (name, analysis[name].to_numpy()) for name in self.covariates
                ],
                y_name=self.y,
                d_name=self.treat,
                covariate_names=self.covariates,
                observation_ids=observation_ids,
                n_obs=len(self.data),
                n_rep=self.n_rep,
                n_folds=self.n_folds,
                analysis_has_missing=bool(analysis.isna().to_numpy().any()),
                fold_has_missing=bool(
                    raw_fold is not None and np.asarray(pd.isna(raw_fold)).any()
                ),
            )
            external_predictions = external_input.predictions
            Y, D, X = external_input.y, external_input.d, external_input.x
            if raw_fold is not None:
                normalized_fold = self._validate_fold_indices(
                    raw_fold, len(self.data), self.n_folds
                )
                validate_external_partitions(external_predictions, normalized_fold)
            if store_oof:
                identity = _retention.external_analysis_identity(
                    predictions=external_predictions,
                    n_input=len(self.data),
                    observation_ids_source=external_input.observation_ids_source,
                )
                collector = _retention.OOFRetention.external(
                    predictions=external_predictions,
                    identity=identity,
                    n_rep=self.n_rep,
                    n_folds=self.n_folds,
                    trimming_threshold=self.trimming_threshold,
                )
                external_predictions = collector._predictions
            Z = sample_weight = fold_indices = None
            fold_source = "external_predictions"
        else:
            cols = [self.y, self.treat] + self.covariates
            if self.instrument is not None:
                cols = cols + self.instrument
            work = self.data[cols].copy()
            sw = self._sample_weight_input
            if isinstance(sw, str):
                work["__sw__"] = self.data[sw].astype(float).values
            elif sw is not None:
                work["__sw__"] = np.asarray(sw, dtype=float)
            fi = self._fold_indices_input
            if isinstance(fi, str):
                work["__fold__"] = self.data[fi].values
            elif fi is not None and np.ndim(fi) == 2:
                if len(self._cluster_cols) != 2:
                    raise MethodIncompatibility(
                        "fold_indices with two columns is the per-dimension "
                        "fold assignment of two-way cluster=; pass a 1-D array "
                        "otherwise."
                    )
                fi2 = np.asarray(fi)
                if fi2.shape != (len(self.data), 2):
                    raise MethodIncompatibility(
                        f"two-way fold_indices must have shape ({len(self.data)}, "
                        f"2); got {fi2.shape}."
                    )
                work["__fold__"] = fi2[:, 0]
                work["__fold1__"] = fi2[:, 1]
            elif fi is not None:
                work["__fold__"] = np.asarray(fi)
            if fi is not None:
                # a missing label on a usable row is a hole in the supplied
                # design, not missing data: listwise deletion would hide it
                usable = work.drop(columns="__fold__").notna().all(axis=1)
                n_unlabelled = int((work["__fold__"].isna() & usable).sum())
                if n_unlabelled:
                    raise MethodIncompatibility(
                        f"fold_indices contain missing values on {n_unlabelled} "
                        "row(s) that are otherwise complete."
                    )
            if self._cluster_cols:
                work["__cl__"] = self.data[self._cluster_cols[0]].values
                if len(self._cluster_cols) == 2:
                    work["__cl1__"] = self.data[self._cluster_cols[1]].values
            if self._design is not None:
                work["__sw__"] = self._design.weights
                work["__svy_row__"] = np.arange(len(work))
                if self._design.cluster_col is not None:
                    work["__cl__"] = self._design._psu_codes
            if store_oof:
                identity = _retention.internal_analysis_identity(
                    n_input=len(self.data),
                    complete_mask=~work.isna().to_numpy().any(axis=1),
                    observation_ids=observation_ids,
                )
            clean = work.dropna()
            if "__svy_row__" in clean.columns:
                self._design_rows = clean["__svy_row__"].to_numpy()
            Y = clean[self.y].values.astype(float)
            D = _treatment_as_float(clean[self.treat], function="dml")
            X = clean[self.covariates].values.astype(float)
            Z = (
                clean[self.instrument[0]].values.astype(float)
                if self.instrument is not None
                else None
            )
            if "__sw__" in clean.columns:
                sample_weight = clean["__sw__"].values.astype(float)
                if np.any(sample_weight < 0):
                    raise MethodIncompatibility(
                        "sample_weight must be non-negative; got negative entries."
                    )
                if not np.isfinite(sample_weight).all():
                    raise MethodIncompatibility(
                        "sample_weight contains non-finite values."
                    )
                if sample_weight.sum() <= 0:
                    raise DataInsufficient("sample_weight has zero total mass.")
            else:
                sample_weight = None
            if "__fold1__" in clean.columns:
                fold_indices = np.column_stack(
                    [
                        self._validate_fold_indices(
                            clean[c].values, len(Y), self.n_folds
                        )
                        for c in ("__fold__", "__fold1__")
                    ]
                )
                fold_source = "user"
            elif "__fold__" in clean.columns:
                fold_indices = self._validate_fold_indices(
                    clean["__fold__"].values,
                    len(Y),
                    self.n_folds,
                )
                fold_source = "user"
            else:
                fold_indices = None
                fold_source = "kfold"
            if "__cl1__" in clean.columns:
                if fold_indices is not None and np.ndim(fold_indices) != 2:
                    raise MethodIncompatibility(
                        "DML two-way cluster=: fold_indices must give the fold "
                        "of each unit in both cluster dimensions (an n x 2 "
                        "array).",
                        recovery_hint="Pass np.column_stack([fold_dim1, "
                        "fold_dim2]) or drop fold_indices.",
                    )
                codes2 = np.column_stack(
                    [pd.factorize(clean[c])[0] for c in ("__cl__", "__cl1__")]
                )
                self._cluster_codes = codes2
                for j in range(2):
                    n_cl = int(codes2[:, j].max()) + 1
                    if n_cl < self.n_folds:
                        raise DataInsufficient(
                            f"DML cluster=: {n_cl} clusters in "
                            f"'{self._cluster_cols[j]}' cannot fill "
                            f"n_folds={self.n_folds} cluster-level folds."
                        )
                    if fold_indices is not None:
                        per_cluster = (
                            pd.Series(fold_indices[:, j])
                            .groupby(codes2[:, j])
                            .nunique()
                        )
                        if (per_cluster > 1).any():
                            raise MethodIncompatibility(
                                "DML two-way cluster=: fold_indices column "
                                f"{j} splits a '{self._cluster_cols[j]}' cluster "
                                "across folds.",
                                diagnostics={
                                    "n_split_clusters": int((per_cluster > 1).sum())
                                },
                            )
                if fold_indices is None:
                    fold_source = "cluster_kfold_twoway"
            elif "__cl__" in clean.columns:
                if fold_indices is not None and np.ndim(fold_indices) == 2:
                    raise MethodIncompatibility(  # pragma: no cover - guarded above
                        "two-column fold_indices need two-way cluster=."
                    )
                self._cluster_codes = pd.factorize(clean["__cl__"])[0]
                n_cl = int(self._cluster_codes.max()) + 1
                if n_cl < self.n_folds:
                    raise DataInsufficient(
                        f"DML cluster=: {n_cl} clusters cannot fill "
                        f"n_folds={self.n_folds} cluster-level folds."
                    )
                if fold_indices is not None:
                    per_cluster = (
                        pd.Series(fold_indices).groupby(self._cluster_codes).nunique()
                    )
                    if (per_cluster > 1).any():
                        raise MethodIncompatibility(
                            "DML cluster=: the supplied fold_indices split a "
                            "cluster across folds; every cluster must lie in "
                            "one fold.",
                            diagnostics={
                                "n_split_clusters": int((per_cluster > 1).sum())
                            },
                        )
                else:
                    fold_source = "cluster_kfold"
            if store_oof:
                collector = _retention.OOFRetention.internal(
                    y=Y,
                    d=D,
                    x=X,
                    covariate_names=self.covariates,
                    identity=identity,
                    n_rep=self.n_rep,
                    n_folds=self.n_folds,
                    random_state=self.random_state,
                    trimming_threshold=self.trimming_threshold,
                )
        n = len(Y)
        if n == 0:
            raise DataInsufficient(
                "DML has no complete rows after dropping missing values."
            )
        if n < self.n_folds:
            raise DataInsufficient(
                f"DML needs at least n_folds complete rows; got n={n}, "
                f"n_folds={self.n_folds}."
            )
        thetas: List[float] = []
        ses: List[float] = []
        per_rep_diags: List[Dict[str, Any]] = []
        last_residuals: Dict[str, np.ndarray] = {}
        sens_reps: List[Dict[str, Any]] = []
        for rep in range(self.n_rep):
            self._last_rep_diagnostics: Dict[str, Any] = {}
            self._last_rep_residuals: Dict[str, np.ndarray] = {}
            self._last_rep_sens: Dict[str, np.ndarray] = {}
            self._last_rep_ar: Dict[str, Any] = {}
            capture = collector and collector.start_rep(rep, self.random_state + rep)
            if capture is not None:
                self._oof_rep_capture = capture
            try:
                if external_predictions is None:
                    theta_r, se_r = self._fit_one_rep(
                        Y,
                        D,
                        X,
                        Z,
                        n,
                        rng_seed=self.random_state + rep,
                        sample_weight=sample_weight,
                        fold_indices=fold_indices,
                    )
                else:
                    theta_r, se_r = self._fit_external_rep(
                        Y, D, X, external_predictions, rep
                    )
            finally:
                self.__dict__.pop("_oof_rep_capture", None)
            if self._design is not None:
                theta_r, se_r = self._design_theta_se(self._last_rep_score)
            elif self._cluster_codes is not None:
                theta_r, se_r = self._cluster_theta_se(self._last_rep_score)
            if collector is not None:
                assert isinstance(capture, _retention.OOFRepCapture)
                collector.finish_rep(capture, theta_r, se_r)
            thetas.append(theta_r)
            ses.append(se_r)
            if self._last_rep_sens and not self._is_twoway():
                # the omitted-variable bound has no two-way cluster variance
                sens_reps.append(self._sensitivity_rep(theta_r, se_r))
            if self._last_rep_diagnostics:
                per_rep_diags.append(self._last_rep_diagnostics)
            if self._last_rep_residuals:
                last_residuals = self._last_rep_residuals
        if len(thetas) == 1:
            theta, se = thetas[0], ses[0]
        else:
            # Median estimate with within-rep variance and split dispersion:
            #     σ̂² = median_r ( se_r² + (θ̂_r − θ̂_med)² )
            thetas_arr = np.asarray(thetas, dtype=float)
            ses_arr = np.asarray(ses, dtype=float)
            theta = float(np.median(thetas_arr))
            s2 = ses_arr**2 + (thetas_arr - theta) ** 2
            se = float(np.sqrt(np.median(s2)))
        retained_bundle = collector and collector.build(theta, se)
        t_stat = theta / se if se > 0 else 0.0
        pvalue = float(2 * stats.norm.sf(abs(t_stat)))
        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci = (theta - z_crit * se, theta + z_crit * se)
        svy_df: Optional[float] = None
        if self._design is not None:
            from ..survey.linearization import design_df, design_inference

            svy_df = design_df(self._design, self._design_rows)
            ci, pvalue = design_inference(theta, se, svy_df, self.alpha)
        model_info = {
            "dml_model": self._MODEL_TAG,
            "n_folds": self.n_folds,
            "n_rep": self.n_rep,
            "ml_g": type(self.ml_g).__name__,
            "ml_m": type(self.ml_m).__name__,
            "default_learners": bool(getattr(self, "_default_learners", False)),
            "n_covariates": len(self.covariates),
            "fold_source": fold_source,
        }
        if model_info["default_learners"] and self._MODEL_TAG == "PLR":
            # The only coverage evidence for this configuration is below
            # nominal, so the result says so where summary() prints it.
            # A warning on every default call would be noise: the evidence
            # is one design, and the docstring gives the full account.
            model_info["learner_note"] = _DEFAULT_PLR_LEARNER_NOTE
        if svy_df is not None:
            from ..survey.linearization import design_info

            model_info["se_method"] = "survey_linearization"
            model_info["survey_design"] = design_info(self._design, svy_df)
        if self.cluster is not None:
            model_info["cluster"] = self.cluster
            if self._cluster_codes is None:
                model_info["n_clusters"] = None
            elif self._is_twoway():
                model_info["n_clusters"] = [
                    int(self._cluster_codes[:, j].max()) + 1 for j in range(2)
                ]
                model_info["n_cells"] = self.n_folds**2
            else:
                model_info["n_clusters"] = int(self._cluster_codes.max()) + 1
        if self.score is not None:
            model_info["score"] = self.score
        if self._USES_IPW:
            model_info["normalize_ipw"] = self.normalize_ipw
            model_info["trimming_threshold"] = self.trimming_threshold
        if self._REQUIRES_INSTRUMENT:
            if self.instrument is None:  # pragma: no cover
                raise MethodIncompatibility(
                    f"dml.{self._MODEL_TAG.lower()}: instrument is required"
                )
            model_info["ml_r"] = type(self.ml_r).__name__
            model_info["instrument"] = self.instrument[0]
        if self.n_rep > 1:
            model_info["theta_all_reps"] = thetas
            model_info["se_all_reps"] = ses
        if per_rep_diags:
            model_info["diagnostics"] = self._aggregate_diagnostics(per_rep_diags)
            self._warn_propensity_clipping(model_info["diagnostics"], n)
        if external_predictions is not None:
            model_info["scoring_engine"] = "statspai_irm_external_predictions"
            model_info["external_predictions_hashes"] = dict(external_input.hashes)
            model_info["nuisance_fit"] = "skipped_external_predictions"
        # Keep private diagnostics in memory for sensitivity helpers.
        if last_residuals:
            model_info["_y_resid"] = last_residuals.get("y_resid")
            model_info["_d_resid"] = last_residuals.get("d_resid")
            model_info["_pscore"] = last_residuals.get("pscore")
            if last_residuals.get("dr_scores") is not None:
                model_info["_dr_scores"] = last_residuals["dr_scores"]
        if sens_reps:
            model_info["_sens"] = sens_reps
        if self._last_rep_ar:
            # Weak-instrument-robust set of the last repetition (PLIV).
            model_info["anderson_rubin"] = self._last_rep_ar
        model_info["_X_design"] = X
        model_info["_T"] = D
        model_info["_Y"] = Y
        model_info["_covariate_names"] = list(self.covariates)
        result = CausalResult(
            method=f"Double ML ({self._MODEL_TAG})",
            estimand=self._ESTIMAND,
            estimate=theta,
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=n,
            detail=None,
            model_info=model_info,
            _citation_key="dml",
        )
        if retained_bundle is not None:
            _attach_result_oof(result, retained_bundle)
        return result
