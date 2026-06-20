"""
Super Learner: Ensemble meta-learner for nuisance parameter estimation.

The Super Learner (van der Laan et al. 2007) finds the optimal weighted
combination of a library of candidate learners via cross-validation:

    weights = argmin_{w >= 0, sum(w)=1} sum_i (y_i - sum_k w_k * f_k(x_i))^2

where ``f_k`` are the out-of-fold predictions of each base learner.

This is a **convex-combination** quadratic program (QP) on the simplex:
non-negativity AND sum-to-one. Earlier versions of this module solved
the unconstrained NNLS problem and rescaled the solution by its sum;
that is *not* the simplex-constrained optimum and produced systematically
biased ensemble predictions. We now solve the QP directly via
:func:`scipy.optimize.minimize` (SLSQP).

References
----------
van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007).
"Super Learner."
Statistical Applications in Genetics and Molecular Biology, 6(1). [@vanderlaan2007super]
"""

from typing import Any, List, Optional, TYPE_CHECKING
import numpy as np
from scipy.optimize import minimize

from ..exceptions import DataInsufficient, MethodIncompatibility

# sklearn is imported lazily inside the methods that need it so that
# ``import statspai`` doesn't pull ~245 sklearn submodules through this
# file when the user never touches super_learner. ``BaseEstimator`` only
# appears in type annotations here and is gated behind ``TYPE_CHECKING``.
if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


# ======================================================================
# Public API
# ======================================================================


def super_learner(
    X: np.ndarray,
    y: np.ndarray,
    library: "Optional[List[BaseEstimator]]" = None,
    n_folds: int = 5,
    task: str = "regression",
    random_state: int = 42,
) -> "SuperLearner":
    """
    Fit a Super Learner ensemble.

    Parameters
    ----------
    X : np.ndarray (n, p)
    y : np.ndarray (n,)
    library : list of sklearn estimators, optional
        Candidate learners. If None, uses a default library.
    n_folds : int, default 5
        Cross-validation folds.
    task : str, default 'regression'
        'regression' or 'classification'.
    random_state : int, default 42

    Returns
    -------
    SuperLearner
        Fitted ensemble.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> from sklearn.linear_model import LinearRegression, Ridge
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> X = rng.normal(size=(n, 3))
    >>> beta = np.array([1.0, -0.5, 0.3])
    >>> y = X @ beta + rng.normal(scale=0.5, size=n)
    >>> sl = sp.super_learner(
    ...     X, y,
    ...     library=[LinearRegression(), Ridge(alpha=1.0)],
    ...     n_folds=3,
    ... )
    >>> round(float(sl.weights_.sum()), 2)  # simplex weights
    1.0
    >>> sl.predict(X[:5]).shape
    (5,)
    """
    sl = SuperLearner(
        library=library, n_folds=n_folds, task=task, random_state=random_state
    )
    sl.fit(X, y)
    return sl


# ======================================================================
# Super Learner class
# ======================================================================


class SuperLearner:
    """
    Super Learner ensemble (van der Laan et al. 2007).

    Parameters
    ----------
    library : list of sklearn estimators, optional
        Candidate learners. If None, uses a default diverse library.
    n_folds : int, default 5
        Number of cross-validation folds.
    task : str, default 'regression'
        'regression' or 'classification'.
    random_state : int, default 42

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> from sklearn.linear_model import LinearRegression, Ridge
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> X = rng.normal(size=(n, 3))
    >>> y = X @ np.array([1.0, -0.5, 0.3]) + rng.normal(scale=0.5, size=n)
    >>> sl = sp.SuperLearner(
    ...     library=[LinearRegression(), Ridge(alpha=1.0)],
    ...     n_folds=3,
    ... ).fit(X, y)
    >>> round(float(sl.weights_.sum()), 2)  # simplex weights
    1.0
    >>> sl.predict(X[:5]).shape
    (5,)

    References
    ----------
    vanderlaan2007super
    """

    def __init__(
        self,
        library: "Optional[List[BaseEstimator]]" = None,
        n_folds: int = 5,
        task: str = "regression",
        random_state: int = 42,
    ):
        self.library = library
        self.n_folds = n_folds
        self.task = task
        self.random_state = random_state
        self._fitted = False
        self.n_features_in_: Optional[int] = None

    def fit(self, X: Any, y: Any) -> "SuperLearner":
        """
        Fit the Super Learner.

        1. Get cross-validated predictions from each base learner.
        2. Find optimal weights via simplex-constrained least squares.
        3. Refit all base learners on full data.
        """
        from sklearn.base import clone
        from sklearn.model_selection import KFold, StratifiedKFold

        self._validate_fit_controls()
        X, y = self._prepare_fit_arrays(X, y)
        n = len(y)

        if self.library is None:
            self.library = self._default_library()
        if not isinstance(self.library, list) or len(self.library) == 0:
            raise MethodIncompatibility(
                "SuperLearner.fit() requires a non-empty learner library.",
                recovery_hint="Pass at least one sklearn-style estimator.",
                diagnostics={"library_type": type(self.library).__name__},
            )
        n_learners = len(self.library)

        # Multiclass guard: classification SuperLearner pulls
        # ``predict_proba(X)[:, 1]`` and would silently drop other class
        # columns for ≥3-class targets. Reject up-front rather than
        # train a learner whose predictions are garbage.
        if self.task == "classification":
            uniq = np.unique(y)
            if len(uniq) > 2:
                raise MethodIncompatibility(
                    "SuperLearner(task='classification') only supports binary "
                    f"targets; got {len(uniq)} unique values: "
                    f"{uniq[:10].tolist()}{'...' if len(uniq) > 10 else ''}. "
                    "Encode multi-class outcomes as binary indicators or use "
                    "task='regression' on a probabilistic target."
                )
            if len(uniq) < 2:
                raise DataInsufficient(
                    "SuperLearner(task='classification') needs two classes.",
                    recovery_hint="Provide both outcome classes.",
                    diagnostics={"classes": uniq.tolist()},
                )
            if not np.array_equal(uniq, np.array([0.0, 1.0])):
                raise MethodIncompatibility(
                    "SuperLearner(task='classification') expects y coded 0/1.",
                    recovery_hint=(
                        "Recode the negative class to 0 and positive class to 1."
                    ),
                    diagnostics={"classes": uniq.tolist()},
                )
            _, class_counts = np.unique(y, return_counts=True)
            if np.min(class_counts) < self.n_folds:
                raise DataInsufficient(
                    "SuperLearner(task='classification') needs at least "
                    "n_folds observations in each class.",
                    recovery_hint=(
                        "Reduce n_folds or provide more observations in the "
                        "rarer class."
                    ),
                    diagnostics={
                        "n_folds": int(self.n_folds),
                        "class_counts": class_counts.tolist(),
                    },
                )

        # Step 1: Cross-validated predictions
        # Stratify on y for classification so every fold contains both
        # classes (essential when one class is rare); the previous plain
        # KFold could put all positives into a single fold and crash
        # downstream `predict_proba` calls on constant-class folds.
        if self.task == "classification":
            splitter = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            split_iter = splitter.split(X, y)
        else:
            splitter = KFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            split_iter = splitter.split(X)

        cv_preds = np.zeros((n, n_learners))

        for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te = X[test_idx]

            for k, learner in enumerate(self.library):
                m = clone(learner)
                m.fit(X_tr, y_tr)
                if self.task == "classification" and hasattr(m, "predict_proba"):
                    cv_preds[test_idx, k] = m.predict_proba(X_te)[:, 1]
                else:
                    cv_preds[test_idx, k] = m.predict(X_te)

        # Step 2: Find optimal simplex-constrained weights.
        # Solve: min_w  ||y - Z w||^2   s.t.  w >= 0,  sum(w) = 1.
        # Earlier versions used NNLS (drops the sum-to-1 constraint) and
        # rescaled the solution; that's the wrong optimum (the rescaled
        # NNLS solution is the simplex optimum only when the
        # unconstrained sum already equals 1, a measure-zero event).
        # We instead solve the QP directly via SLSQP on the squared
        # loss, which is convex with a unique global minimum.
        Z = cv_preds
        ZTZ = Z.T @ Z
        ZTy = Z.T @ y

        def _obj(w: np.ndarray) -> float:
            # 0.5 * ||y - Z w||² up to a constant
            return 0.5 * float(w @ ZTZ @ w) - float(ZTy @ w)

        def _grad(w: np.ndarray) -> np.ndarray:
            return np.asarray(ZTZ @ w - ZTy)

        w0 = np.ones(n_learners) / n_learners
        bounds = [(0.0, 1.0)] * n_learners
        constraints = (
            {
                "type": "eq",
                "fun": lambda w: float(np.sum(w) - 1.0),
                "jac": lambda w: np.ones_like(w),
            },
        )
        res = minimize(
            _obj,
            w0,
            jac=_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 200},
        )
        if res.success:
            weights = np.clip(res.x, 0.0, 1.0)
            # Renormalise tiny round-off drift so sum == 1 exactly.
            s = weights.sum()
            weights = weights / s if s > 0 else np.ones(n_learners) / n_learners
        else:
            # Fall back to equal weights; surface the failure so
            # callers can investigate. Equal weights are still a valid
            # ensemble — just not the optimum.
            import warnings

            warnings.warn(
                f"SuperLearner: simplex QP failed to converge "
                f"({res.message!r}); falling back to equal weights.",
                UserWarning,
                stacklevel=2,
            )
            weights = np.ones(n_learners) / n_learners

        self.weights_ = weights
        self.cv_preds_ = cv_preds

        # CV risk per learner
        self.cv_risks_ = np.array(
            [np.mean((y - cv_preds[:, k]) ** 2) for k in range(n_learners)]
        )

        # Step 3: Refit all learners on full data
        self._fitted_learners = []
        for learner in self.library:
            m = clone(learner)
            m.fit(X, y)
            self._fitted_learners.append(m)

        self.n_features_in_ = X.shape[1]
        self._fitted = True
        return self

    def _validate_fit_controls(self) -> None:
        """Validate scalar controls before cross-validation starts."""
        if self.task not in {"regression", "classification"}:
            raise MethodIncompatibility(
                "SuperLearner.fit(): task must be 'regression' or " "'classification'.",
                recovery_hint="Choose task='regression' or task='classification'.",
                diagnostics={"task": self.task},
            )
        if (
            not isinstance(self.n_folds, (int, np.integer))
            or isinstance(self.n_folds, bool)
            or int(self.n_folds) < 2
        ):
            raise MethodIncompatibility(
                "SuperLearner.fit(): n_folds must be an integer >= 2.",
                recovery_hint="Use at least two cross-validation folds.",
                diagnostics={"n_folds": self.n_folds},
            )
        self.n_folds = int(self.n_folds)

    def _prepare_fit_arrays(self, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        """Coerce and validate fit arrays with clear taxonomy errors."""
        try:
            X_arr = np.asarray(X, dtype=np.float64)
            y_arr = np.asarray(y, dtype=np.float64).ravel()
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "SuperLearner.fit() requires numeric X and y.",
                recovery_hint="Convert features and target to numeric arrays.",
            ) from exc
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        elif X_arr.ndim != 2:
            raise MethodIncompatibility(
                "SuperLearner.fit(): X must be a 1D or 2D numeric array.",
                recovery_hint="Pass X shaped (n_samples, n_features).",
                diagnostics={"x_ndim": int(X_arr.ndim)},
            )
        if X_arr.shape[0] != y_arr.shape[0]:
            raise MethodIncompatibility(
                "SuperLearner.fit(): X and y must have the same row count.",
                recovery_hint="Align features and target to the same sample.",
                diagnostics={
                    "n_x": int(X_arr.shape[0]),
                    "n_y": int(y_arr.shape[0]),
                },
            )
        if X_arr.shape[0] < self.n_folds:
            raise DataInsufficient(
                "SuperLearner.fit() needs at least n_folds observations.",
                recovery_hint="Reduce n_folds or provide more observations.",
                diagnostics={"nobs": int(X_arr.shape[0]), "n_folds": self.n_folds},
            )
        if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise DataInsufficient(
                "SuperLearner.fit() needs at least one row and one feature.",
                recovery_hint="Pass a non-empty feature matrix.",
                diagnostics={
                    "nobs": int(X_arr.shape[0]),
                    "n_features": int(X_arr.shape[1]),
                },
            )
        if not np.isfinite(X_arr).all():
            raise MethodIncompatibility(
                "SuperLearner.fit(): X contains NaN or infinite values.",
                recovery_hint="Drop or impute non-finite feature rows.",
            )
        if not np.isfinite(y_arr).all():
            raise MethodIncompatibility(
                "SuperLearner.fit(): y contains NaN or infinite values.",
                recovery_hint="Drop or impute non-finite target rows.",
            )
        return X_arr, y_arr

    def _prepare_predict_matrix(self, X: Any) -> np.ndarray:
        """Validate prediction features against the fitted feature count."""
        try:
            X_arr = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "SuperLearner.predict() requires numeric X.",
                recovery_hint="Convert prediction features to numeric values.",
            ) from exc
        expected = self.n_features_in_
        if X_arr.ndim == 1:
            if expected is not None and expected > 1 and X_arr.size == expected:
                X_arr = X_arr.reshape(1, -1)
            elif expected in (None, 1):
                X_arr = X_arr.reshape(-1, 1)
            else:
                raise MethodIncompatibility(
                    "SuperLearner.predict(): one-dimensional input does not "
                    "match the fitted feature count.",
                    recovery_hint="Pass X shaped (n_samples, n_features).",
                    diagnostics={
                        "n_values": int(X_arr.size),
                        "expected_features": expected,
                    },
                )
        elif X_arr.ndim != 2:
            raise MethodIncompatibility(
                "SuperLearner.predict(): X must be a 1D or 2D numeric array.",
                recovery_hint="Pass X shaped (n_samples, n_features).",
                diagnostics={"x_ndim": int(X_arr.ndim)},
            )
        if X_arr.shape[0] == 0:
            raise DataInsufficient(
                "SuperLearner.predict() received no rows.",
                recovery_hint="Pass at least one prediction row.",
            )
        if expected is not None and X_arr.shape[1] != expected:
            raise MethodIncompatibility(
                "SuperLearner.predict(): feature count does not match fit().",
                recovery_hint=(
                    "Use the same number and order of features used at fit time."
                ),
                diagnostics={
                    "expected_features": expected,
                    "observed_features": int(X_arr.shape[1]),
                },
            )
        if not np.isfinite(X_arr).all():
            raise MethodIncompatibility(
                "SuperLearner.predict(): X contains NaN or infinite values.",
                recovery_hint="Drop or impute non-finite prediction rows.",
            )
        return X_arr

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict using the weighted ensemble.

        For classification (``task='classification'``) the returned
        values are the convex combination of base-learner probability
        predictions and are clipped to ``(1e-6, 1 - 1e-6)`` so that
        callers can take ``logit(.)`` without inf. For regression no
        clipping is applied.

        Parameters
        ----------
        X : np.ndarray (n, p)

        Returns
        -------
        np.ndarray (n,)
        """
        if not self._fitted:
            raise MethodIncompatibility(
                "SuperLearner.predict() requires a fitted ensemble.",
                recovery_hint="Call fit() before predict().",
            )

        X = self._prepare_predict_matrix(X)
        n = X.shape[0]
        n_learners = len(self._fitted_learners)

        preds = np.zeros((n, n_learners))
        for k, m in enumerate(self._fitted_learners):
            if self.task == "classification" and hasattr(m, "predict_proba"):
                preds[:, k] = m.predict_proba(X)[:, 1]
            else:
                preds[:, k] = m.predict(X)

        out = preds @ self.weights_
        # Classification: clip ensemble output away from {0, 1} so
        # downstream logit / inverse-propensity callers can't blow up.
        # `predict_proba` used to clip but `predict` did not — that
        # asymmetry meant TMLE callers using `sl_g.predict(W)` could
        # get exact 0/1 if every base learner produced exact 0/1
        # (e.g. RandomForest with deterministic terminal nodes).
        if self.task == "classification":
            out = np.clip(out, 1e-6, 1 - 1e-6)
        return np.asarray(out)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict probabilities (for classification task).

        Identical to :meth:`predict` under ``task='classification'`` —
        kept as a separate method for sklearn-style API parity.
        """
        return self.predict(X)

    def summary(self) -> str:
        """Print Super Learner summary."""
        if not self._fitted:
            raise ValueError("SuperLearner must be fitted first.")

        lines = []
        lines.append("Super Learner Ensemble")
        lines.append("-" * 50)
        lines.append(f"{'Learner':<30} {'Weight':>8} {'CV Risk':>10}")
        lines.append("-" * 50)

        assert self.library is not None
        for k, learner in enumerate(self.library):
            name = type(learner).__name__
            w = self.weights_[k]
            risk = self.cv_risks_[k]
            marker = " *" if w > 0.01 else ""
            lines.append(f"{name:<30} {w:>8.4f} {risk:>10.6f}{marker}")

        lines.append("-" * 50)
        return "\n".join(lines)

    def _default_library(self) -> List[Any]:
        """Build a diverse default library of learners."""
        from sklearn.linear_model import (
            LinearRegression,
            Ridge,
            Lasso,
            LogisticRegression,
        )
        from sklearn.ensemble import (
            RandomForestRegressor,
            RandomForestClassifier,
            GradientBoostingRegressor,
            GradientBoostingClassifier,
        )
        from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

        if self.task == "classification":
            return [
                LogisticRegression(max_iter=1000, random_state=self.random_state),
                RandomForestClassifier(
                    n_estimators=100, max_depth=5, random_state=self.random_state
                ),
                GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.random_state,
                ),
                KNeighborsClassifier(n_neighbors=10),
            ]
        else:
            return [
                LinearRegression(),
                Ridge(alpha=1.0),
                Lasso(alpha=0.1, max_iter=5000),
                RandomForestRegressor(
                    n_estimators=100, max_depth=5, random_state=self.random_state
                ),
                GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.random_state,
                ),
                KNeighborsRegressor(n_neighbors=10),
            ]
