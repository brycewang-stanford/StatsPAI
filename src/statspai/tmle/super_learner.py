"""
Super Learner: Ensemble meta-learner for nuisance parameter estimation.

The Super Learner (van der Laan et al. 2007) finds the optimal weighted
combination of a library of candidate learners via cross-validation.

    weights = argmin_{w >= 0, sum(w)=1} sum_i (y_i - sum_k w_k * f_k(x_i))^2

where f_k are the cross-validated predictions of each base learner.

This is solved as a non-negative least squares problem (NNLS).

References
----------
van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007).
"Super Learner."
Statistical Applications in Genetics and Molecular Biology, 6(1).
"""

from typing import Optional, List, Dict, Any
import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import KFold
from scipy.optimize import nnls


# ======================================================================
# Public API
# ======================================================================

def super_learner(
    X: np.ndarray,
    y: np.ndarray,
    library: Optional[List[BaseEstimator]] = None,
    n_folds: int = 5,
    task: str = 'regression',
    random_state: int = 42,
) -> 'SuperLearner':
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
    """
    sl = SuperLearner(library=library, n_folds=n_folds,
                      task=task, random_state=random_state)
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
    """

    def __init__(
        self,
        library: Optional[List[BaseEstimator]] = None,
        n_folds: int = 5,
        task: str = 'regression',
        random_state: int = 42,
    ):
        self.library = library
        self.n_folds = n_folds
        self.task = task
        self.random_state = random_state
        self._fitted = False

    def fit(self, X, y):
        """
        Fit the Super Learner.

        1. Get cross-validated predictions from each base learner.
        2. Find optimal weights via NNLS.
        3. Refit all base learners on full data.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)

        if self.library is None:
            self.library = self._default_library()

        n_learners = len(self.library)

        # Step 1: Cross-validated predictions
        kf = KFold(n_splits=self.n_folds, shuffle=True,
                    random_state=self.random_state)

        cv_preds = np.zeros((n, n_learners))

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te = X[test_idx]

            for k, learner in enumerate(self.library):
                m = clone(learner)
                m.fit(X_tr, y_tr)
                if self.task == 'classification' and hasattr(m, 'predict_proba'):
                    cv_preds[test_idx, k] = m.predict_proba(X_te)[:, 1]
                else:
                    cv_preds[test_idx, k] = m.predict(X_te)

        # Step 2: Find optimal weights via NNLS
        # Solve: min ||y - Z @ w||^2 s.t. w >= 0
        weights, _ = nnls(cv_preds, y)

        # Normalise to sum to 1
        w_sum = weights.sum()
        if w_sum > 0:
            weights = weights / w_sum
        else:
            weights = np.ones(n_learners) / n_learners

        self.weights_ = weights
        self.cv_preds_ = cv_preds

        # CV risk per learner
        self.cv_risks_ = np.array([
            np.mean((y - cv_preds[:, k]) ** 2) for k in range(n_learners)
        ])

        # Step 3: Refit all learners on full data
        self._fitted_learners = []
        for learner in self.library:
            m = clone(learner)
            m.fit(X, y)
            self._fitted_learners.append(m)

        self._fitted = True
        return self

    def predict(self, X):
        """
        Predict using the weighted ensemble.

        Parameters
        ----------
        X : np.ndarray (n, p)

        Returns
        -------
        np.ndarray (n,)
        """
        if not self._fitted:
            raise ValueError("SuperLearner must be fitted first.")

        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        n_learners = len(self._fitted_learners)

        preds = np.zeros((n, n_learners))
        for k, m in enumerate(self._fitted_learners):
            if self.task == 'classification' and hasattr(m, 'predict_proba'):
                preds[:, k] = m.predict_proba(X)[:, 1]
            else:
                preds[:, k] = m.predict(X)

        return preds @ self.weights_

    def predict_proba(self, X):
        """Predict probabilities (for classification task)."""
        p = self.predict(X)
        return np.clip(p, 1e-6, 1 - 1e-6)

    def summary(self) -> str:
        """Print Super Learner summary."""
        if not self._fitted:
            raise ValueError("SuperLearner must be fitted first.")

        lines = []
        lines.append("Super Learner Ensemble")
        lines.append("-" * 50)
        lines.append(f"{'Learner':<30} {'Weight':>8} {'CV Risk':>10}")
        lines.append("-" * 50)

        for k, learner in enumerate(self.library):
            name = type(learner).__name__
            w = self.weights_[k]
            risk = self.cv_risks_[k]
            marker = " *" if w > 0.01 else ""
            lines.append(f"{name:<30} {w:>8.4f} {risk:>10.6f}{marker}")

        lines.append("-" * 50)
        return "\n".join(lines)

    def _default_library(self):
        """Build a diverse default library of learners."""
        from sklearn.linear_model import (
            LinearRegression, Ridge, Lasso, LogisticRegression,
        )
        from sklearn.ensemble import (
            RandomForestRegressor, RandomForestClassifier,
            GradientBoostingRegressor, GradientBoostingClassifier,
        )
        from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

        if self.task == 'classification':
            return [
                LogisticRegression(max_iter=1000, random_state=self.random_state),
                RandomForestClassifier(n_estimators=100, max_depth=5,
                                       random_state=self.random_state),
                GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                           learning_rate=0.1,
                                           random_state=self.random_state),
                KNeighborsClassifier(n_neighbors=10),
            ]
        else:
            return [
                LinearRegression(),
                Ridge(alpha=1.0),
                Lasso(alpha=0.1, max_iter=5000),
                RandomForestRegressor(n_estimators=100, max_depth=5,
                                      random_state=self.random_state),
                GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                          learning_rate=0.1,
                                          random_state=self.random_state),
                KNeighborsRegressor(n_neighbors=10),
            ]
