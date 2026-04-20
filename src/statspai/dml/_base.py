"""
Shared infrastructure for Double/Debiased ML estimators.

Each model-specific file (``plr.py``, ``irm.py``, ``pliv.py``,
``iivm.py``) inherits from :class:`_DoubleMLBase` and supplies its own
Neyman-orthogonal score via ``_fit_one_rep``. The base class handles
validation, default learners, repeat-split aggregation, and
:class:`CausalResult` construction.
"""

from typing import Optional, List, Any, Union
import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


class _DoubleMLBase:
    """Abstract base: common plumbing for all DML estimators."""

    # Overridden by subclasses
    _MODEL_TAG: str = ''            # short label, used in method= string
    _ESTIMAND: str = 'ATE'          # 'ATE' or 'LATE'
    _REQUIRES_INSTRUMENT: bool = False
    _BINARY_TREATMENT: bool = False  # True → default_ml_m is classifier
    _BINARY_INSTRUMENT: bool = False  # True → default_ml_r is classifier

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
    ):
        self.data = data
        self.y = y
        self.treat = treat
        self.covariates = list(covariates)
        if instrument is None:
            self.instrument = None
        elif isinstance(instrument, str):
            self.instrument = [instrument]
        else:
            self.instrument = list(instrument)
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.alpha = alpha

        self._validate()

        self.ml_g = ml_g if ml_g is not None else self._default_ml_g()
        self.ml_m = ml_m if ml_m is not None else self._default_ml_m()
        self.ml_r = ml_r if ml_r is not None else self._default_ml_r()

    def _validate(self):
        required = [self.y, self.treat] + self.covariates
        if self.instrument is not None:
            required = required + self.instrument
        for col in required:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        if self._REQUIRES_INSTRUMENT and not self.instrument:
            raise ValueError(
                f"model='{self._MODEL_TAG.lower()}' requires an "
                f"'instrument' argument"
            )
        if not self._REQUIRES_INSTRUMENT and self.instrument is not None:
            raise ValueError(
                f"'instrument' is only valid when model requires an IV "
                f"(got model='{self._MODEL_TAG.lower()}')"
            )
        if self._REQUIRES_INSTRUMENT and self.instrument is not None and len(self.instrument) > 1:
            raise ValueError(
                f"model='{self._MODEL_TAG.lower()}' accepts a single scalar "
                f"instrument; got {len(self.instrument)}: {self.instrument}. "
                f"For multiple instruments, project them onto a scalar index "
                f"and pass that column name."
            )
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")

    def _default_ml_g(self):
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42,
        )

    def _default_ml_m(self):
        if self._BINARY_TREATMENT:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42,
            )
        return self._default_ml_g()

    def _default_ml_r(self):
        if self._BINARY_INSTRUMENT:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42,
            )
        return self._default_ml_g()

    # Subclasses implement this: return (theta, se) for ONE rep.
    def _fit_one_rep(self, Y, D, X, Z, n, rng_seed):
        raise NotImplementedError

    def fit(self) -> CausalResult:
        """Cross-fit, aggregate across repeats, return a CausalResult."""
        cols = [self.y, self.treat] + self.covariates
        if self.instrument is not None:
            cols = cols + self.instrument
        clean = self.data[cols].dropna()
        Y = clean[self.y].values.astype(float)
        D = clean[self.treat].values.astype(float)
        X = clean[self.covariates].values.astype(float)
        Z = (
            clean[self.instrument[0]].values.astype(float)
            if self.instrument is not None else None
        )
        n = len(Y)

        thetas: List[float] = []
        ses: List[float] = []
        for rep in range(self.n_rep):
            theta, se = self._fit_one_rep(Y, D, X, Z, n, rng_seed=42 + rep)
            thetas.append(theta)
            ses.append(se)

        if len(thetas) == 1:
            theta, se = thetas[0], ses[0]
        else:
            theta = float(np.median(thetas))
            se = float(np.median(ses))

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
        }
        if self._REQUIRES_INSTRUMENT:
            model_info['ml_r'] = type(self.ml_r).__name__
            model_info['instrument'] = self.instrument[0]
        if self.n_rep > 1:
            model_info['theta_all_reps'] = thetas
            model_info['se_all_reps'] = ses

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
