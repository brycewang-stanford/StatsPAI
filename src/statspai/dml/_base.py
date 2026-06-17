"""
Shared infrastructure for Double/Debiased ML estimators.

Each model-specific file (``plr.py``, ``irm.py``, ``pliv.py``,
``iivm.py``) inherits from :class:`_DoubleMLBase` and supplies its own
Neyman-orthogonal score via ``_fit_one_rep``. The base class handles
validation, default learners, repeat-split aggregation, and
:class:`CausalResult` construction.
"""

import operator
from typing import Optional, List, Any, Union
import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._learners import resolve_learner


def _positive_int(value: Any, *, name: str, context: str) -> int:
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise MethodIncompatibility(
            f"{context}: {name} must be a positive integer"
        ) from exc
    if isinstance(value, bool) or parsed < 1:
        raise MethodIncompatibility(
            f"{context}: {name} must be a positive integer"
        )
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


class _DoubleMLBase:
    """Abstract base: common plumbing for all DML estimators."""

    # Overridden by subclasses
    _MODEL_TAG: str = ''            # short label, used in method= string
    _ESTIMAND: str = 'ATE'          # 'ATE' or 'LATE'
    _REQUIRES_INSTRUMENT: bool = False
    # Whether ``ml_m`` / ``ml_r`` model a binary target, i.e. should
    # default to a classifier and accept binary learner aliases. Naming
    # caveat: in PLR / IRM the ml_m target is D (treatment); in IIVM it
    # is Z (instrument propensity). Hence the target-shape name, not
    # ``_BINARY_TREATMENT`` — these are nuisance-target descriptors,
    # not estimand descriptors.
    _ML_M_TARGET_BINARY: bool = False
    _ML_R_TARGET_BINARY: bool = False
    # Some IV models (IIVM) genuinely only work with a single scalar
    # instrument; PLIV with multiple Z is fine in principle but the
    # current reduced-form r(X) is scalar so we still project to a
    # scalar index before passing. Models that can handle vector Z
    # override this to False.
    _REQUIRES_SCALAR_INSTRUMENT: bool = True
    # Subclasses opt into ``sample_weight`` support by setting this to
    # True. Models without weighted-variance derivations (PLIV, IIVM)
    # raise ``NotImplementedError`` if a non-trivial weight vector is
    # supplied — better than silently ignoring it.
    _SUPPORTS_SAMPLE_WEIGHT: bool = False

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
            covariates, name="covariates", context=context,
        )
        if instrument is None:
            self.instrument = None
        elif isinstance(instrument, str):
            self.instrument = [instrument]
        else:
            self.instrument = _coerce_column_list(
                instrument, name="instrument", context=context,
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
        if fold_indices is not None and self._MODEL_TAG != "PLR":
            raise MethodIncompatibility(
                f"{context}: explicit fold_indices are currently supported for "
                "model='plr' only."
            )
        if fold_indices is not None and self.n_rep != 1:
            raise MethodIncompatibility(
                f"{context}: explicit fold_indices require n_rep=1; pass one fold "
                "assignment for the single cross-fit repetition."
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
        else:
            arr = np.asarray(fold_indices)
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
        if (
            self._sample_weight_input is not None
            and not self._SUPPORTS_SAMPLE_WEIGHT
        ):
            raise MethodIncompatibility(  # pragma: no cover
                f"{context}: sample_weight is not yet supported for "
                f"model='{self._MODEL_TAG.lower()}'. Weighted support is "
                f"currently implemented for model in "
                f"{{'plr', 'irm', 'pliv', 'iivm'}} only."
            )

        self._validate()

        self.ml_g = (
            self._default_ml_g() if ml_g is None
            else resolve_learner(ml_g, kind="regressor", role="ml_g")
        )
        self.ml_m = (
            self._default_ml_m() if ml_m is None
            else resolve_learner(
                ml_m,
                kind="classifier" if self._ML_M_TARGET_BINARY else "regressor",
                role="ml_m",
            )
        )
        self.ml_r = (
            self._default_ml_r() if ml_r is None
            else resolve_learner(
                ml_r,
                kind="classifier" if self._ML_R_TARGET_BINARY else "regressor",
                role="ml_r",
            )
        )

    def _validate(self):
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

    def _default_ml_g(self):
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42,
        )

    def _default_ml_m(self):
        if self._ML_M_TARGET_BINARY:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42,
            )
        return self._default_ml_g()

    def _default_ml_r(self):
        if self._ML_R_TARGET_BINARY:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
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
        self, Y, D, X, Z, n, rng_seed, sample_weight=None, fold_indices=None
    ):
        raise NotImplementedError  # pragma: no cover

    @staticmethod
    def _validate_fold_indices(fold_indices, n: int, n_folds: int) -> np.ndarray:
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
                f"fold_indices define {len(unique)} folds, but n_folds="
                f"{n_folds}"
            )
        counts = np.bincount(codes, minlength=n_folds)
        if np.any(counts == 0):
            raise DataInsufficient(
                "fold_indices must assign at least one row per fold"
            )
        return codes.astype(int)

    # ----- Sample-weight helpers (used by subclasses) -----------------
    @staticmethod
    def _fit_weighted(learner, X, y, weights):
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

    def fit(self) -> CausalResult:
        """Cross-fit, aggregate across repeats, return a CausalResult."""
        cols = [self.y, self.treat] + self.covariates
        if self.instrument is not None:
            cols = cols + self.instrument
        # Build a working frame that also carries sample weights so the
        # dropna mask is consistent across (Y, D, X, Z, w).
        work = self.data[cols].copy()
        sw = self._sample_weight_input
        if isinstance(sw, str):
            work["__sw__"] = self.data[sw].astype(float).values
        elif sw is not None:
            work["__sw__"] = np.asarray(sw, dtype=float)
        fi = self._fold_indices_input
        if isinstance(fi, str):
            work["__fold__"] = self.data[fi].values
        elif fi is not None:
            work["__fold__"] = np.asarray(fi)
        clean = work.dropna()
        Y = clean[self.y].values.astype(float)
        D = clean[self.treat].values.astype(float)
        X = clean[self.covariates].values.astype(float)
        Z = (
            clean[self.instrument[0]].values.astype(float)
            if self.instrument is not None else None
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
        if "__fold__" in clean.columns:
            fold_indices = self._validate_fold_indices(
                clean["__fold__"].values, n, self.n_folds,
            )
            fold_source = "user"
        else:
            fold_indices = None
            fold_source = "kfold"

        thetas: List[float] = []
        ses: List[float] = []
        per_rep_diags: List[dict] = []
        last_residuals: dict = {}
        for rep in range(self.n_rep):
            self._last_rep_diagnostics = {}
            self._last_rep_residuals = {}
            theta_r, se_r = self._fit_one_rep(
                Y, D, X, Z, n, rng_seed=self.random_state + rep,
                sample_weight=sample_weight,
                fold_indices=fold_indices,
            )
            thetas.append(theta_r)
            ses.append(se_r)
            if self._last_rep_diagnostics:
                per_rep_diags.append(self._last_rep_diagnostics)
            if self._last_rep_residuals:
                last_residuals = self._last_rep_residuals

        if len(thetas) == 1:
            theta, se = thetas[0], ses[0]
        else:
            # Chernozhukov et al. (2018) eq. 3.7 / Algorithm 1 Step 4:
            # point estimate = median of rep estimates,
            # SE accounts for BOTH within-rep nuisance variance AND
            # between-rep dispersion of the point estimates:
            #     σ̂² = median_r ( se_r² + (θ̂_r − θ̂_med)² )
            # This avoids under-coverage that would result from
            # taking only median(se_r).
            thetas_arr = np.asarray(thetas, dtype=float)
            ses_arr = np.asarray(ses, dtype=float)
            theta = float(np.median(thetas_arr))
            s2 = ses_arr**2 + (thetas_arr - theta) ** 2
            se = float(np.sqrt(np.median(s2)))

        t_stat = theta / se if se > 0 else 0.0
        pvalue = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci = (theta - z_crit * se, theta + z_crit * se)

        model_info = {
            'dml_model': self._MODEL_TAG,
            'n_folds': self.n_folds,
            'n_rep': self.n_rep,
            'ml_g': type(self.ml_g).__name__,
            'ml_m': type(self.ml_m).__name__,
            'n_covariates': len(self.covariates),
            'fold_source': fold_source,
        }
        if self._REQUIRES_INSTRUMENT:
            model_info['ml_r'] = type(self.ml_r).__name__
            model_info['instrument'] = self.instrument[0]
        if self.n_rep > 1:
            model_info['theta_all_reps'] = thetas
            model_info['se_all_reps'] = ses
        if per_rep_diags:
            model_info['diagnostics'] = self._aggregate_diagnostics(per_rep_diags)
        # Stash residuals + design matrix for downstream sensitivity /
        # diagnostics (sp.dml_sensitivity, sp.dml_diagnostics). These are
        # NumPy arrays so they don't serialise in to_dict, but they're
        # available on the in-memory model_info.
        if last_residuals:
            model_info.update({
                "_y_resid": last_residuals.get("y_resid"),
                "_d_resid": last_residuals.get("d_resid"),
                "_pscore": last_residuals.get("pscore"),
            })
        model_info["_X_design"] = X
        model_info["_T"] = D
        model_info["_Y"] = Y
        model_info["_covariate_names"] = list(self.covariates)

        return CausalResult(
            method=f'Double ML ({self._MODEL_TAG})',
            estimand=self._ESTIMAND,
            estimate=theta,
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=n,
            detail=None,
            model_info=model_info,
            _citation_key='dml',
        )
