"""
Causal Forest implementation for heterogeneous treatment effect estimation

This module implements the Causal Forest algorithm for estimating conditional
average treatment effects (CATE) based on the methodology from:
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous
  treatment effects using random forests. Journal of the American Statistical
  Association, 113(523), 1228-1242.

The implementation is inspired by and partially based on the EconML library:
- https://github.com/py-why/econml/
- Microsoft Corporation. (2019). EconML: A Python Package for ML-Based
  Heterogeneous Treatment Effects Estimation.

Key features:
- Honest random forests for unbiased treatment effect estimation
- Bootstrap confidence intervals
- Compatible with StatsPAI outreg2 export functionality
- Both formula and direct array interfaces
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeRegressor
import warnings

# Import our core classes
from ..core.base import BaseModel
from ..exceptions import DataInsufficient, MethodIncompatibility


class CausalForest(BaseModel):
    """
    Causal Forest for heterogeneous treatment effect estimation

    This class implements the Causal Forest algorithm, which uses random forests
    to estimate conditional average treatment effects (CATE) in a non-parametric way.

    The method combines ideas from:
    1. Honest estimation to avoid overfitting
    2. Double machine learning to handle confounding
    3. Random forests for flexible function approximation

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest
    min_samples_leaf : int, default=5
        Minimum number of samples required to be at a leaf node
    max_depth : int, default=None
        Maximum depth of trees
    max_samples : float, default=0.5
        Fraction of samples to use for each tree
    model_y : estimator, optional
        Model for outcome regression (first stage)
    model_t : estimator, optional
        Model for treatment propensity (first stage)
    discrete_treatment : bool, default=True
        Whether treatment is discrete (binary/categorical) or continuous
    honest : bool, default=True
        Whether to use honest estimation (separate samples for splitting and effects)
    bootstrap : bool, default=True
        Whether to use bootstrap sampling for trees
    random_state : int, optional
        Random state for reproducibility
    n_jobs : int, default=1
        Number of parallel jobs
    verbose : int, default=0
        Verbosity level

    Attributes
    ----------
    fitted_ : bool
        Whether the model has been fitted
    params : pd.Series
        Not applicable for non-parametric methods, returns empty Series
    std_errors : pd.Series
        Not applicable for non-parametric methods, returns empty Series
    tvalues : pd.Series
        Not applicable for non-parametric methods, returns empty Series
    pvalues : np.ndarray
        Not applicable for non-parametric methods, returns empty array
    diagnostics : dict
        Model diagnostics and fit statistics
    data_info : dict
        Information about the data used in fitting

    Notes
    -----
    This implementation is inspired by the EconML library's CausalForestDML
    but adapted to fit the StatsPAI architecture and interface.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from statspai.forest import CausalForest
    >>>
    >>> # Generate sample data
    >>> np.random.seed(42)
    >>> n = 1000
    >>> X = np.random.normal(0, 1, (n, 3))
    >>> T = np.random.binomial(1, 0.5, n)
    >>> Y = X[:, 0] * T + X[:, 1] + np.random.normal(0, 1, n)
    >>> data = pd.DataFrame({
    ...     'Y': Y, 'T': T, 'X1': X[:, 0], 'X2': X[:, 1], 'X3': X[:, 2]
    ... })
    >>>
    >>> # Fit Causal Forest
    >>> cf = CausalForest(n_estimators=50, random_state=42)
    >>> cf.fit('Y ~ T | X1 + X2 + X3', data=data)
    >>>
    >>> # Estimate treatment effects
    >>> cate = cf.effect(data[['X1', 'X2', 'X3']])
    >>> print(f"Average treatment effect: {cate.mean():.3f}")
    """

    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_leaf: int = 5,
        max_depth: Optional[int] = None,
        max_samples: float = 0.5,
        model_y: Optional[BaseEstimator] = None,
        model_t: Optional[BaseEstimator] = None,
        discrete_treatment: bool = True,
        honest: bool = True,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.model_y: BaseEstimator
        self.model_t: BaseEstimator
        self.discrete_treatment = discrete_treatment
        self.honest = honest
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Initialize default models if not provided
        if model_y is None:
            self.model_y = RandomForestRegressor(
                n_estimators=100, random_state=random_state
            )
        else:
            self.model_y = model_y
        if model_t is not None:
            self.model_t = model_t
        else:
            if discrete_treatment:
                self.model_t = RandomForestClassifier(
                    n_estimators=100, random_state=random_state
                )
            else:
                self.model_t = RandomForestRegressor(
                    n_estimators=100, random_state=random_state
                )

        # Initialize internal state
        self.fitted_ = False
        self._forest: Optional[List[DecisionTreeRegressor]] = None
        self._treatment_values: Optional[np.ndarray] = None
        self._feature_names: Optional[List[str]] = None

    def fit(  # type: ignore[override]
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        Y: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        W: Optional[np.ndarray] = None,
    ) -> "CausalForest":
        """
        Fit the Causal Forest model

        Parameters
        ----------
        formula : str, optional
            Formula specification in the form "Y ~ T | X1 + X2 + ... [| W1 + W2 + ...]"
            where Y is outcome, T is treatment, X are effect modifiers, W are controls
        data : pd.DataFrame, optional
            Data containing all variables if using formula interface
        Y : array-like, optional
            Outcome variable (n_samples,)
        T : array-like, optional
            Treatment variable (n_samples,)
        X : array-like, optional
            Effect modifier variables (n_samples, n_features)
        W : array-like, optional
            Control variables for confounding adjustment (n_samples, n_controls)

        Returns
        -------
        self : CausalForest
            Fitted estimator
        """
        self._validate_fit_controls()

        # Parse inputs
        if formula is not None and data is not None:
            Y, T, X, W = self._parse_formula_inputs(formula, data)
        elif Y is not None and T is not None and X is not None:
            self._feature_names = None
            Y, T, X, W = self._validate_array_inputs(Y, T, X, W)
        else:
            raise ValueError(
                "Must provide either (formula, data) or (Y, T, X) arguments"
            )

        # Validate inputs
        try:
            Y = np.asarray(Y, dtype=float).ravel()
            T = np.asarray(T, dtype=float).ravel()
            X = np.asarray(X, dtype=float)
            if W is not None:
                W = np.asarray(W, dtype=float)
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "CausalForest.fit() requires numeric Y, T, X, and W inputs.",
                recovery_hint=(
                    "Convert the outcome, treatment, effect modifiers, and "
                    "controls to numeric columns before fitting."
                ),
            ) from exc

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise MethodIncompatibility(
                "CausalForest.fit(): X must be a 1D or 2D numeric array.",
                recovery_hint="Pass X shaped (n_samples, n_features).",
                diagnostics={"x_ndim": int(X.ndim)},
            )
        if W is not None and W.ndim == 1:
            W = W.reshape(-1, 1)
        elif W is not None and W.ndim != 2:
            raise MethodIncompatibility(
                "CausalForest.fit(): W must be a 1D or 2D numeric array.",
                recovery_hint="Pass W shaped (n_samples, n_controls).",
                diagnostics={"w_ndim": int(W.ndim)},
            )

        n_samples = len(Y)
        if len(T) != n_samples or len(X) != n_samples:
            raise MethodIncompatibility(
                "CausalForest.fit(): Y, T, and X must have the same row count.",
                recovery_hint="Align all fit inputs to the same sample.",
                diagnostics={
                    "n_y": int(len(Y)),
                    "n_t": int(len(T)),
                    "n_x": int(len(X)),
                },
            )
        if W is not None and len(W) != n_samples:
            raise MethodIncompatibility(
                "CausalForest.fit(): W must have the same row count as Y.",
                recovery_hint="Align controls to the outcome sample.",
                diagnostics={"n_y": int(len(Y)), "n_w": int(len(W))},
            )
        if n_samples < 3:
            raise DataInsufficient(
                "CausalForest.fit() needs at least 3 rows for cross-fitting.",
                recovery_hint="Use at least 3 complete observations.",
                diagnostics={"nobs": int(n_samples)},
            )
        if X.shape[1] == 0:
            raise MethodIncompatibility(
                "CausalForest.fit(): X must contain at least one feature.",
                recovery_hint="Pass at least one effect modifier column.",
            )
        if not np.isfinite(Y).all():
            raise MethodIncompatibility(
                "CausalForest.fit(): outcome contains NaN or infinite values.",
                recovery_hint="Drop or impute non-finite outcome rows.",
            )
        if not np.isfinite(T).all():
            raise MethodIncompatibility(
                "CausalForest.fit(): treatment contains NaN or infinite values.",
                recovery_hint="Drop or impute non-finite treatment rows.",
            )
        if not np.isfinite(X).all():
            raise MethodIncompatibility(
                "CausalForest.fit(): X contains NaN or infinite values.",
                recovery_hint="Drop or impute non-finite effect modifiers.",
            )
        if W is not None and not np.isfinite(W).all():
            raise MethodIncompatibility(
                "CausalForest.fit(): W contains NaN or infinite values.",
                recovery_hint="Drop or impute non-finite controls.",
            )
        n_tree_samples = int(float(self.max_samples) * n_samples)
        min_tree_samples = 2 if self.honest else 1
        if n_tree_samples < min_tree_samples:
            raise DataInsufficient(
                "CausalForest.fit(): max_samples leaves too few rows per tree.",
                recovery_hint=(
                    "Increase max_samples or fit on more observations so each "
                    "tree has enough rows."
                ),
                diagnostics={
                    "nobs": int(n_samples),
                    "max_samples": float(self.max_samples),
                    "rows_per_tree": int(n_tree_samples),
                    "minimum_rows_per_tree": min_tree_samples,
                },
            )

        # Store data info
        self.data_info = {
            "nobs": n_samples,
            "n_features": X.shape[1],
            "n_controls": W.shape[1] if W is not None else 0,
            "treatment_values": np.unique(T),
        }

        treatment_values = np.unique(T)
        self._treatment_values = treatment_values
        self.data_info["treatment_values"] = treatment_values
        if self._feature_names is None or len(self._feature_names) != X.shape[1]:
            self._feature_names = [f"X{i}" for i in range(X.shape[1])]
        self._X_original = X.copy()
        self._T_original = T.copy()
        self._Y_original = Y.copy()

        # Validate treatment
        if self.discrete_treatment:
            if len(treatment_values) < 2:
                raise DataInsufficient(
                    "CausalForest.fit() needs two treatment values.",
                    recovery_hint="Provide both treated and control observations.",
                    diagnostics={"treatment_values": treatment_values.tolist()},
                )
            if len(treatment_values) > 2:
                raise MethodIncompatibility(
                    "CausalForest.fit() currently supports binary treatment only.",
                    recovery_hint=(
                        "Use sp.multi_arm_forest() for multi-arm treatments or "
                        "fit separate binary contrasts."
                    ),
                    diagnostics={"treatment_values": treatment_values.tolist()},
                    alternative_functions=["sp.multi_arm_forest"],
                )
            if not np.array_equal(treatment_values, np.array([0.0, 1.0])):
                raise MethodIncompatibility(
                    "CausalForest.fit() expects binary treatment coded as 0/1.",
                    recovery_hint="Recode the control group to 0 and treated group to 1.",
                    diagnostics={"treatment_values": treatment_values.tolist()},
                )
            _, treatment_counts = np.unique(T, return_counts=True)
            if np.min(treatment_counts) < 3:
                raise DataInsufficient(
                    "CausalForest.fit() needs at least 3 observations per "
                    "treatment group for 3-fold cross-fitting.",
                    recovery_hint=(
                        "Add observations or reduce the estimator to a design "
                        "that does not require 3-fold nuisance cross-fitting."
                    ),
                    diagnostics={
                        "treatment_values": treatment_values.tolist(),
                        "counts": treatment_counts.tolist(),
                    },
                )
        elif np.var(T) <= 1e-12:
            raise DataInsufficient(
                "CausalForest.fit() needs treatment variation.",
                recovery_hint="Provide a non-constant continuous treatment.",
            )

        # Step 1: Fit first stage models (Double ML approach)
        # This follows the EconML CausalForestDML implementation
        if self.verbose > 0:
            print("Fitting first stage models...")

        # Prepare features for first stage
        first_stage_features = X if W is None else np.hstack([X, W])

        # Fit outcome model
        if self.verbose > 0:
            print("  Fitting outcome model...")
        self.model_y.fit(first_stage_features, Y)
        Y_pred = cross_val_predict(
            self.model_y, first_stage_features, Y, cv=3, n_jobs=self.n_jobs
        )
        Y_residual = Y - Y_pred

        # Fit treatment model
        if self.verbose > 0:
            print("  Fitting treatment model...")
        self.model_t.fit(first_stage_features, T)
        if self.discrete_treatment:
            T_pred = cross_val_predict(
                self.model_t,
                first_stage_features,
                T,
                cv=3,
                method="predict_proba",
                n_jobs=self.n_jobs,
            )
            if T_pred.ndim == 2 and T_pred.shape[1] == 2:
                # Binary case - use probability of positive class (class 1)
                T_pred = T_pred[:, 1]
                T_residual = T - T_pred
            elif T_pred.ndim == 1 or T_pred.shape[1] == 1:
                # Single class case
                T_pred = T_pred.ravel()
                T_residual = T - T_pred
            else:
                # Multi-class case - use one-hot encoding
                T_onehot = np.zeros((len(T), len(self._treatment_values)))
                for i, val in enumerate(self._treatment_values):
                    T_onehot[T == val, i] = 1
                T_residual = T_onehot - T_pred
        else:
            T_pred = cross_val_predict(
                self.model_t, first_stage_features, T, cv=3, n_jobs=self.n_jobs
            )
            T_residual = T - T_pred

        # Stash cross-fitted nuisance predictions for downstream inference
        # (used by :func:`forest_inference.calibration_test` / :func:`rate`).
        self._m_insample = Y_pred
        self._e_insample = T_pred if np.asarray(T_pred).ndim == 1 else T_pred[:, 0]

        # Step 2: Fit causal forest on residuals
        if self.verbose > 0:
            print("Fitting causal forest...")

        self._forest = self._fit_causal_forest(X, T_residual, Y_residual)

        # Mark as fitted
        self.fitted_ = True

        # Store results for compatibility with EconometricResults
        # For non-parametric methods, we don't have traditional parameters
        self.params = pd.Series([], dtype=float)
        self.std_errors = pd.Series([], dtype=float)
        self.tvalues = pd.Series([], dtype=float)
        self.pvalues = np.array([])

        # Compute diagnostics
        ate = self.effect(X).mean()
        self.diagnostics = {
            "method": "Causal Forest",
            "n_estimators": self.n_estimators,
            "n_features": X.shape[1],
            "average_treatment_effect": ate,
            "treatment_type": "discrete" if self.discrete_treatment else "continuous",
        }

        return self

    def _parse_formula_inputs(
        self, formula: str, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Parse formula inputs to extract Y, T, X, W arrays"""
        # Parse formula: "Y ~ T | X1 + X2 + ... [| W1 + W2 + ...]"
        parts = formula.split("|")
        if len(parts) < 2:
            raise ValueError(
                "Formula must have format 'Y ~ T | X1 + X2 + ...' or "
                "'Y ~ T | X1 + X2 + ... | W1 + W2 + ...'"
            )

        # Parse outcome and treatment
        yt_part = parts[0].strip()
        if "~" not in yt_part:
            raise ValueError(
                "Formula must contain '~' to separate outcome and treatment"
            )

        y_name, t_name = yt_part.split("~")
        y_name = y_name.strip()
        t_name = t_name.strip()

        # Parse effect modifiers (X)
        x_part = parts[1].strip()
        x_names = [name.strip() for name in x_part.split("+")]

        # Parse controls (W) if provided
        w_names = []
        if len(parts) > 2:
            w_part = parts[2].strip()
            w_names = [name.strip() for name in w_part.split("+")]

        # Extract data
        try:
            Y = data[y_name].values
            T = data[t_name].values
            X = data[x_names].values
            W = data[w_names].values if w_names else None
        except KeyError as e:
            raise ValueError(f"Variable {e} not found in data")

        # Store feature names for later use
        self._feature_names = x_names

        return Y, T, X, W

    def _validate_array_inputs(
        self, Y: np.ndarray, T: np.ndarray, X: np.ndarray, W: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Validate direct array inputs"""
        Y = np.asarray(Y)
        T = np.asarray(T)
        X = np.asarray(X)
        if W is not None:
            W = np.asarray(W)
        return Y, T, X, W

    def _validate_fit_controls(self) -> None:
        """Validate scalar controls before fitting expensive nuisance models."""
        if (
            not isinstance(self.n_estimators, (int, np.integer))
            or isinstance(self.n_estimators, bool)
            or int(self.n_estimators) < 1
        ):
            raise MethodIncompatibility(
                "CausalForest.fit(): n_estimators must be a positive integer.",
                recovery_hint="Use n_estimators >= 1.",
                diagnostics={"n_estimators": self.n_estimators},
            )
        if (
            not isinstance(self.min_samples_leaf, (int, np.integer))
            or isinstance(self.min_samples_leaf, bool)
            or int(self.min_samples_leaf) < 1
        ):
            raise MethodIncompatibility(
                "CausalForest.fit(): min_samples_leaf must be a positive integer.",
                recovery_hint="Use min_samples_leaf >= 1.",
                diagnostics={"min_samples_leaf": self.min_samples_leaf},
            )
        if self.max_depth is not None and (
            not isinstance(self.max_depth, (int, np.integer))
            or isinstance(self.max_depth, bool)
            or int(self.max_depth) < 1
        ):
            raise MethodIncompatibility(
                "CausalForest.fit(): max_depth must be None or a positive integer.",
                recovery_hint="Use max_depth=None or max_depth >= 1.",
                diagnostics={"max_depth": self.max_depth},
            )
        try:
            max_samples = float(self.max_samples)
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "CausalForest.fit(): max_samples must be a finite fraction.",
                recovery_hint="Use a scalar max_samples in the interval (0, 1].",
                diagnostics={"max_samples": self.max_samples},
            ) from exc
        if not np.isfinite(max_samples) or not 0.0 < max_samples <= 1.0:
            raise MethodIncompatibility(
                "CausalForest.fit(): max_samples must be in the interval (0, 1].",
                recovery_hint="Use a scalar max_samples such as 0.5 or 1.0.",
                diagnostics={"max_samples": self.max_samples},
            )

    def _prepare_effect_matrix(self, data: Any, *, context: str) -> np.ndarray:
        """Validate prediction/effect inputs against the fitted feature schema."""
        expected = None
        if hasattr(self, "data_info"):
            expected = int(self.data_info.get("n_features", 0))
        feature_names = list(getattr(self, "_feature_names", []) or [])

        if isinstance(data, pd.DataFrame):
            if feature_names:
                missing = [name for name in feature_names if name not in data.columns]
                if missing:
                    raise MethodIncompatibility(
                        f"{context}: missing effect-modifier column(s) {missing}.",
                        recovery_hint=(
                            "Pass a DataFrame containing the same effect "
                            "modifier columns used at fit time."
                        ),
                        diagnostics={"missing_columns": missing},
                    )
                selected = data[feature_names]
            else:
                selected = data.select_dtypes(include=[np.number])
                if selected.shape[1] == 0:
                    raise MethodIncompatibility(
                        f"{context}: no numeric effect-modifier columns found.",
                        recovery_hint=(
                            "Pass numeric effect modifiers or fit with a formula "
                            "so column names can be reused."
                        ),
                    )
            try:
                X = selected.to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise MethodIncompatibility(
                    f"{context}: effect-modifier columns must be numeric.",
                    recovery_hint=(
                        "Convert prediction columns to numeric values before "
                        "calling predict() or effect()."
                    ),
                    diagnostics={"columns": list(selected.columns)},
                ) from exc
        else:
            try:
                X = np.asarray(data, dtype=float)
            except (TypeError, ValueError) as exc:
                raise MethodIncompatibility(
                    f"{context}: effect modifiers must be numeric.",
                    recovery_hint=(
                        "Pass a numeric array or a DataFrame with the fitted "
                        "effect-modifier columns."
                    ),
                ) from exc

        if X.ndim == 1:
            if expected is not None and expected > 1 and X.size == expected:
                X = X.reshape(1, -1)
            elif expected in (None, 0, 1):
                X = X.reshape(-1, 1)
            else:
                raise MethodIncompatibility(
                    f"{context}: one-dimensional input has {X.size} values, "
                    f"but the model was fit with {expected} features.",
                    recovery_hint=(
                        "Pass a two-dimensional array with one row per "
                        "prediction sample."
                    ),
                    diagnostics={
                        "n_values": int(X.size),
                        "expected_features": expected,
                    },
                )
        elif X.ndim != 2:
            raise MethodIncompatibility(
                f"{context}: effect modifiers must be a 1D or 2D array.",
                recovery_hint="Pass an array shaped (n_samples, n_features).",
                diagnostics={"ndim": int(X.ndim)},
            )

        if X.shape[0] == 0:
            raise DataInsufficient(
                f"{context}: no prediction rows were supplied.",
                recovery_hint="Pass at least one row of effect modifiers.",
            )
        if expected not in (None, 0) and X.shape[1] != expected:
            raise MethodIncompatibility(
                f"{context}: expected {expected} effect-modifier column(s), "
                f"got {X.shape[1]}.",
                recovery_hint=(
                    "Use the same number and order of effect modifiers used "
                    "when fitting the causal forest."
                ),
                diagnostics={
                    "expected_features": expected,
                    "observed_features": int(X.shape[1]),
                },
            )
        if not np.isfinite(X).all():
            raise MethodIncompatibility(
                f"{context}: effect modifiers contain NaN or infinite values.",
                recovery_hint=(
                    "Drop or impute non-finite prediction rows before calling "
                    "predict() or effect()."
                ),
            )
        return np.asarray(X, dtype=float)

    def _fit_causal_forest(
        self, X: np.ndarray, T_residual: np.ndarray, Y_residual: np.ndarray
    ) -> List[DecisionTreeRegressor]:
        """
        Fit the causal forest using honest estimation

        This is the core of the causal forest algorithm, implementing honest
        random forests as described in Wager & Athey (2018).
        """
        n_samples, n_features = X.shape
        trees = []

        # Set random seed for reproducibility
        rng = np.random.RandomState(self.random_state)

        for tree_idx in range(self.n_estimators):
            if self.verbose > 1 and tree_idx % 50 == 0:
                print(f"  Fitting tree {tree_idx + 1}/{self.n_estimators}")

            # Sample for this tree
            if self.bootstrap:
                n_tree_samples = int(self.max_samples * n_samples)
                tree_indices = rng.choice(n_samples, n_tree_samples, replace=True)
            else:
                n_tree_samples = int(self.max_samples * n_samples)
                tree_indices = rng.choice(n_samples, n_tree_samples, replace=False)

            # Honest estimation: split samples for tree building and effect estimation
            if self.honest:
                split_idx = len(tree_indices) // 2
                build_indices = tree_indices[:split_idx]
                estimate_indices = tree_indices[split_idx:]
            else:
                build_indices = tree_indices
                estimate_indices = tree_indices

            # Build tree structure using building sample
            # For simplicity, we use sklearn's DecisionTreeRegressor
            # but replace leaf predictions with causal effect estimates
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.randint(0, 2**31),
                max_features="sqrt" if n_features > 1 else None,
            )

            # Fit tree on building sample
            if len(build_indices) > 0:
                # Use treatment residual as target for tree structure
                # Ensure T_residual is 1D for tree fitting
                T_build = T_residual[build_indices]
                if T_build.ndim > 1:
                    T_build = T_build.ravel()

                tree.fit(X[build_indices], T_build)

                # Replace leaf values with honest causal effect estimates
                self._replace_leaf_values_with_causal_effects(
                    tree,
                    X[estimate_indices],
                    T_residual[estimate_indices],
                    Y_residual[estimate_indices],
                )
            else:
                # Fallback if not enough samples
                T_tree = T_residual[tree_indices]
                if T_tree.ndim > 1:
                    T_tree = T_tree.ravel()
                tree.fit(X[tree_indices], T_tree)

            trees.append(tree)

        return trees

    def _replace_leaf_values_with_causal_effects(
        self,
        tree: DecisionTreeRegressor,
        X_estimate: np.ndarray,
        T_residual_estimate: np.ndarray,
        Y_residual_estimate: np.ndarray,
    ) -> None:
        """
        Replace leaf values with honest causal effect estimates

        This implements the honest estimation procedure where leaf values
        are computed using a separate sample from tree construction.
        """
        if len(X_estimate) == 0:
            return

        # Get leaf assignments for estimation sample
        leaf_ids = tree.apply(X_estimate)

        # For each leaf, compute causal effect estimate
        for leaf_id in np.unique(leaf_ids):
            leaf_mask = leaf_ids == leaf_id
            if np.sum(leaf_mask) < 2:  # Need at least 2 samples
                continue

            # Get samples in this leaf
            t_leaf = T_residual_estimate[leaf_mask]
            y_leaf = Y_residual_estimate[leaf_mask]

            # Estimate causal effect in this leaf
            if self.discrete_treatment:
                if t_leaf.ndim == 1:
                    # Binary treatment
                    if np.var(t_leaf) > 1e-8:  # Check for variation
                        # OLS slope on residualized treatment.  Use one
                        # consistent denominator; mixing np.cov's ddof=1
                        # with np.var's ddof=0 inflates small leaves.
                        t_centered = t_leaf - np.mean(t_leaf)
                        y_centered = y_leaf - np.mean(y_leaf)
                        causal_effect = np.sum(y_centered * t_centered) / np.sum(
                            t_centered**2
                        )
                    else:
                        causal_effect = 0.0
                else:
                    # Multi-class treatment - use first treatment effect
                    if np.var(t_leaf[:, 0]) > 1e-8:
                        t_centered = t_leaf[:, 0] - np.mean(t_leaf[:, 0])
                        y_centered = y_leaf - np.mean(y_leaf)
                        causal_effect = np.sum(y_centered * t_centered) / np.sum(
                            t_centered**2
                        )
                    else:
                        causal_effect = 0.0
            else:
                # Continuous treatment
                if np.var(t_leaf) > 1e-8:
                    t_centered = t_leaf - np.mean(t_leaf)
                    y_centered = y_leaf - np.mean(y_leaf)
                    causal_effect = np.sum(y_centered * t_centered) / np.sum(
                        t_centered**2
                    )
                else:
                    causal_effect = 0.0

            # Update THIS leaf's value (leaf_id is the node index in tree.tree_)
            original_shape = tree.tree_.value[leaf_id].shape
            tree.tree_.value[leaf_id] = np.full(original_shape, causal_effect)

    def effect(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate conditional average treatment effects

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Effect modifier variables

        Returns
        -------
        effects : array-like, shape (n_samples,)
            Estimated conditional average treatment effects
        """
        if not self.fitted_:
            raise MethodIncompatibility(
                "CausalForest.effect() requires a fitted model.",
                recovery_hint="Call fit() before requesting treatment effects.",
            )

        X = self._prepare_effect_matrix(X, context="effect()")
        forest = self._forest
        if forest is None:
            raise MethodIncompatibility(
                "CausalForest.effect() has no fitted forest.",
                recovery_hint="Refit the causal forest before requesting effects.",
            )

        # Average predictions across all trees
        predictions_list: List[np.ndarray] = []
        for tree in forest:
            pred = tree.predict(X)
            # Ensure pred is 1D for each tree
            if pred.ndim > 1:
                pred = pred.ravel()
            predictions_list.append(pred)

        # Return average effect across trees
        predictions = np.array(predictions_list)  # shape: (n_trees, n_samples)
        return np.asarray(np.mean(predictions, axis=0), dtype=float)

    def predict(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate treatment effect predictions (required by BaseModel)

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data containing effect modifier variables. If None, uses training data.

        Returns
        -------
        np.ndarray
            Predicted treatment effects

        Notes
        -----
        This method is required by the BaseModel interface. For Causal Forest,
        "predictions" are treatment effect estimates rather than outcome predictions.
        """
        if not self.fitted_:
            raise MethodIncompatibility(
                "CausalForest.predict() requires a fitted model.",
                recovery_hint="Call fit() before requesting predictions.",
            )

        if data is None:
            if hasattr(self, "_X_original"):
                X = self._X_original
            else:
                raise MethodIncompatibility(
                    "CausalForest.predict() has no training data to reuse.",
                    recovery_hint=(
                        "Pass prediction data explicitly or refit the model "
                        "with retained training data."
                    ),
                )
        else:
            X = self._prepare_effect_matrix(data, context="predict()")

        return self.effect(X)

    def effect_interval(
        self, X: np.ndarray, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute confidence intervals for treatment effects using bootstrap

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Effect modifier variables
        alpha : float, default=0.05
            Significance level (1-alpha is confidence level)

        Returns
        -------
        lower : array-like, shape (n_samples,)
            Lower bounds of confidence intervals
        upper : array-like, shape (n_samples,)
            Upper bounds of confidence intervals
        """
        if not self.fitted_:
            raise MethodIncompatibility(
                "CausalForest.effect_interval() requires a fitted model.",
                recovery_hint="Call fit() before requesting intervals.",
            )
        if not np.isscalar(alpha) or not np.isfinite(float(alpha)):
            raise MethodIncompatibility(
                "effect_interval(): alpha must be a finite scalar.",
                recovery_hint="Pass a scalar alpha in the open interval (0, 1).",
            )
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise MethodIncompatibility(
                "effect_interval(): alpha must be in the open interval (0, 1).",
                recovery_hint="Use a confidence level such as alpha=0.05.",
                diagnostics={"alpha": alpha},
            )

        X = self._prepare_effect_matrix(X, context="effect_interval()")
        forest = self._forest
        if forest is None:
            raise MethodIncompatibility(
                "CausalForest.effect_interval() has no fitted forest.",
                recovery_hint="Refit the causal forest before requesting intervals.",
            )

        # Use bootstrap of little bags approach
        # Collect predictions from each tree (which are already bootstrap samples)
        predictions_list: List[np.ndarray] = []
        for tree in forest:
            pred = tree.predict(X)
            # Ensure pred is 1D
            if pred.ndim > 1:
                pred = pred.ravel()
            predictions_list.append(pred)

        # Convert to array and ensure consistent shapes
        try:
            predictions = np.array(predictions_list)  # shape: (n_trees, n_samples)
        except ValueError:
            # Handle inconsistent shapes by ensuring all predictions have same length
            min_len = min(len(p) for p in predictions_list)
            predictions = np.array([p[:min_len] for p in predictions_list])

        # Compute percentiles across trees for each sample
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        lower = np.percentile(predictions, lower_percentile, axis=0)
        upper = np.percentile(predictions, upper_percentile, axis=0)

        return lower, upper

    def average_treatment_effect(
        self,
        X: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        target_sample: str = "all",
        alpha: float = 0.05,
        clip: float = 0.01,
    ) -> Dict[str, float]:
        """GRF-style ATE/ATT/ATC/ATO aggregation of CATE predictions."""
        from .forest_inference import average_treatment_effect

        return average_treatment_effect(
            self,
            X=X,
            T=T,
            target_sample=target_sample,
            alpha=alpha,
            clip=clip,
        )

    def forest_diagnostics(
        self,
        X: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        propensity_bounds: Tuple[float, float] = (0.05, 0.95),
    ) -> Dict[str, object]:
        """Overlap and CATE-distribution diagnostics for this fitted forest."""
        from .forest_inference import forest_diagnostics

        return forest_diagnostics(
            self,
            X=X,
            T=T,
            propensity_bounds=propensity_bounds,
        )

    def summary(self) -> str:
        """
        Return a summary of the fitted model

        Returns
        -------
        summary : str
            Model summary string
        """
        if not self.fitted_:
            return "Causal Forest (not fitted)"

        ate = self.diagnostics.get("average_treatment_effect", 0)

        summary_lines = [
            "=" * 60,
            "Causal Forest Results",
            "=" * 60,
            "Method:                   Causal Forest",
            f"Number of trees:          {self.n_estimators}",
            f"Min samples per leaf:     {self.min_samples_leaf}",
            f"Max depth:                {self.max_depth or 'None'}",
            f"Max samples per tree:     {self.max_samples}",
            f"Honest estimation:        {self.honest}",
            f"Treatment type:           {self.diagnostics.get('treatment_type', 'Unknown')}",
            "",
            f"Number of observations:   {self.data_info.get('nobs', 'Unknown')}",
            f"Number of features:       {self.data_info.get('n_features', 'Unknown')}",
            f"Number of controls:       {self.data_info.get('n_controls', 0)}",
            "",
            f"Average Treatment Effect: {ate:.6f}",
            "=" * 60,
            "",
            "Note: Use .effect(X) to estimate individual treatment effects",
            "      Use .effect_interval(X) for confidence intervals",
        ]

        return "\n".join(summary_lines)

    def __str__(self) -> str:
        """String representation"""
        if self.fitted_:
            return f"CausalForest(fitted=True, n_estimators={self.n_estimators})"
        else:
            return f"CausalForest(fitted=False, n_estimators={self.n_estimators})"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__()

    # ------------------------------------------------------------------ #
    #  GRF-inspired extensions
    # ------------------------------------------------------------------ #

    def variable_importance(self) -> pd.Series:
        """Permutation-based variable importance for the causal forest.

        For each feature j, shuffle its column in the effect-modifier
        matrix and measure how much the cross-validated CATE predictions
        degrade (MSE increase). Higher degradation → more important for
        treatment-effect heterogeneity.

        Returns a normalised importance score (sums to 1).
        """
        if not self.fitted_:
            raise MethodIncompatibility(
                "CausalForest.variable_importance() requires a fitted model.",
                recovery_hint="Call fit() before computing variable importance.",
            )
        X = self._X_original.copy()
        cate_baseline = self.effect(X)
        n, k = X.shape
        rng = np.random.default_rng(0)
        importance = np.empty(k)
        for j in range(k):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            cate_perm = self.effect(X_perm)
            importance[j] = float(np.mean((cate_baseline - cate_perm) ** 2))
        total = importance.sum()
        importance = importance / total if total > 0 else importance
        names = self._feature_names or [f"X{j}" for j in range(k)]
        return pd.Series(importance, index=names).sort_values(ascending=False)

    def best_linear_projection(
        self,
        X_test: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        clip: float = 0.01,
    ) -> pd.DataFrame:
        r"""Best Linear Projection (BLP) of CATE on features (Semenova-Chernozhukov 2021).

        Constructs the augmented inverse-propensity-weighted (AIPW) doubly-robust
        score :math:`\Gamma_i` and regresses it on :math:`X_i` with HC1 standard
        errors:

        .. math::
            \Gamma_i = \hat{\tau}(X_i)
                + \frac{T_i - \hat{e}(X_i)}{\hat{e}(X_i)(1-\hat{e}(X_i))}
                  \bigl(Y_i - \hat{m}(X_i) - (T_i - \hat{e}(X_i))\hat{\tau}(X_i)\bigr).

        :math:`\Gamma_i` is unbiased for :math:`\tau(X_i)` under standard
        cross-fitting / overlap conditions; OLS of :math:`\Gamma_i` on
        :math:`(1, X_i)` recovers the population BLP coefficients with
        valid heteroscedasticity-robust inference (HC1).

        This replaces the earlier plug-in OLS of :math:`\hat{\tau}(X_i)` on
        :math:`X_i`, which produces anti-conservative SEs (the SE on a fitted
        model is not the SE on the population BLP).

        Parameters
        ----------
        X_test : array-like, optional
            Features to evaluate the BLP at; defaults to in-sample X.
        alpha : float, default=0.05
            Significance level for CIs (reported alongside coef/SE).
        clip : float, default=0.01
            Propensity clip (binary discrete treatment only) to prevent the
            inverse-propensity term from blowing up under near-violations of
            overlap. Counts of clipped units are exposed via ``self.diagnostics``.

        Returns
        -------
        pd.DataFrame
            Index ``["Intercept", *features]`` with columns
            ``[coef, se, t, p, ci_lower, ci_upper]``. HC1 SEs.

        References
        ----------
        Semenova V., Chernozhukov V. (2021).
        "Debiased Machine Learning of Conditional Average Treatment Effects
        and Other Causal Functions." *Econometrics Journal* 24(2): 264-289.
        DOI: 10.1093/ectj/utaa027.
        """
        if not self.fitted_:
            raise MethodIncompatibility(
                "CausalForest.best_linear_projection() requires a fitted model.",
                recovery_hint="Call fit() before computing the BLP.",
            )
        try:
            alpha_value = float(alpha)
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "best_linear_projection(): alpha must be a finite scalar.",
                recovery_hint="Use an alpha value in the open interval (0, 1).",
                diagnostics={"alpha": alpha},
            ) from exc
        if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
            raise MethodIncompatibility(
                "best_linear_projection(): alpha must be in the open interval (0, 1).",
                recovery_hint="Use an alpha value such as 0.05.",
                diagnostics={"alpha": alpha},
            )
        try:
            clip_value = float(clip)
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "best_linear_projection(): clip must be a finite scalar.",
                recovery_hint="Use a clip value in the interval [0, 0.5).",
                diagnostics={"clip": clip},
            ) from exc
        if not np.isfinite(clip_value) or not 0.0 <= clip_value < 0.5:
            raise MethodIncompatibility(
                "best_linear_projection(): clip must be in the interval [0, 0.5).",
                recovery_hint="Use a small propensity clip such as 0.01.",
                diagnostics={"clip": clip},
            )
        from scipy import stats as _stats

        if X_test is None:
            X = self._X_original
        else:
            X = self._prepare_effect_matrix(
                X_test,
                context="best_linear_projection()",
            )
        n, k = X.shape

        # Use the in-sample DR construction (requires Y, T, m̂, ê).
        if X_test is not None and X.shape[0] != self._Y_original.shape[0]:
            warnings.warn(
                "best_linear_projection: X_test has a different sample size "
                "than the training data; falling back to the plug-in CATE "
                "regression with HC1 SE (no DR construction possible "
                "out-of-sample).",
                UserWarning,
                stacklevel=2,
            )
            cate = self.effect(X)
            gamma = cate
        else:
            tau = self.effect(self._X_original)
            Y = self._Y_original.astype(float)
            T = self._T_original.astype(float)
            m_hat = np.asarray(self._m_insample, dtype=float)
            e_hat = np.asarray(self._e_insample, dtype=float)

            if self.discrete_treatment:
                e_clip = np.clip(e_hat, clip_value, 1.0 - clip_value)
                n_clipped = int(np.sum((e_hat < clip_value) | (e_hat > 1 - clip_value)))
                weight = (T - e_clip) / (e_clip * (1.0 - e_clip))
            else:
                # Continuous treatment: use partialled-out Robinson form.
                # Γ_i = τ̂(X_i) + (T_i - ê(X_i))(Y_i - m̂(X_i) - (T_i-ê)τ̂)/Var(T-ê)
                resid_T = T - e_hat
                var_T = float(np.mean(resid_T**2)) + 1e-12
                weight = resid_T / var_T
                n_clipped = 0

            mu_hat = m_hat + (T - e_hat) * tau
            gamma = tau + weight * (Y - mu_hat)

            self.diagnostics = dict(self.diagnostics or {})
            self.diagnostics["blp_n_clipped_propensities"] = n_clipped

        # Regress Γ_i on (1, X_std) with HC1 SE.
        X_mean = X.mean(axis=0)
        X_sd = X.std(axis=0) + 1e-12
        X_std = (X - X_mean) / X_sd
        D = np.column_stack([np.ones(n), X_std])
        DtD = D.T @ D
        try:
            DtD_inv = np.linalg.inv(DtD)
        except np.linalg.LinAlgError:
            DtD_inv = np.linalg.pinv(DtD)
        beta = DtD_inv @ (D.T @ gamma)
        resid = gamma - D @ beta

        # HC1 White SE with (n / (n - p)) finite-sample correction.
        p = D.shape[1]
        omega = D * resid[:, None]
        meat = omega.T @ omega
        cov = DtD_inv @ meat @ DtD_inv * (n / max(n - p, 1))
        se = np.sqrt(np.diag(cov))
        tvals = beta / np.where(se > 0, se, np.nan)
        df = max(n - p, 1)
        pvals = 2.0 * (1.0 - _stats.t.cdf(np.abs(tvals), df=df))
        z = float(_stats.t.ppf(1.0 - alpha_value / 2.0, df=df))

        names = ["Intercept"] + (self._feature_names or [f"X{j}" for j in range(k)])
        return pd.DataFrame(
            {
                "coef": beta,
                "se": se,
                "t": tvals,
                "p": pvals,
                "ci_lower": beta - z * se,
                "ci_upper": beta + z * se,
            },
            index=names,
        )

    def ate(self, X: Optional[np.ndarray] = None) -> float:
        """Average Treatment Effect (mean CATE)."""
        if not self.fitted_:
            raise MethodIncompatibility(
                "CausalForest.ate() requires a fitted model.",
                recovery_hint="Call fit() before computing ATE.",
            )
        return float(self.effect(X if X is not None else self._X_original).mean())

    def att(
        self, X: Optional[np.ndarray] = None, T: Optional[np.ndarray] = None
    ) -> float:
        """Average Treatment Effect on the Treated."""
        if not self.fitted_:
            raise MethodIncompatibility(
                "CausalForest.att() requires a fitted model.",
                recovery_hint="Call fit() before computing ATT.",
            )
        if T is None:
            if hasattr(self, "_T_original"):
                T = self._T_original
            else:
                raise MethodIncompatibility(
                    "CausalForest.att() requires a treatment vector.",
                    recovery_hint="Pass T or fit the model before calling att().",
                )
        X = X if X is not None else self._X_original
        cate = self.effect(X)
        try:
            T_arr = np.asarray(T, dtype=float).ravel()
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "CausalForest.att() requires numeric treatment values.",
                recovery_hint="Pass a numeric binary treatment vector.",
            ) from exc
        if T_arr.shape[0] != cate.shape[0]:
            raise MethodIncompatibility(
                "CausalForest.att(): T must have the same row count as X.",
                recovery_hint="Pass treatment values aligned with the CATE rows.",
                diagnostics={
                    "n_t": int(T_arr.shape[0]),
                    "n_effects": int(cate.shape[0]),
                },
            )
        if not np.isfinite(T_arr).all():
            raise MethodIncompatibility(
                "CausalForest.att() requires finite treatment values.",
                recovery_hint="Drop or impute rows with NaN or infinite treatments.",
            )
        mask = T_arr == 1
        if not np.any(mask):
            raise DataInsufficient(
                "CausalForest.att() received no treated observations.",
                recovery_hint="Pass a treatment vector with at least one T == 1 row.",
            )
        return float(cate[mask].mean())


def causal_forest(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    Y: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    W: Optional[np.ndarray] = None,
    n_estimators: int = 100,
    min_samples_leaf: int = 5,
    max_depth: Optional[int] = None,
    max_samples: float = 0.5,
    model_y: Optional[BaseEstimator] = None,
    model_t: Optional[BaseEstimator] = None,
    discrete_treatment: bool = True,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> CausalForest:
    """
    Convenience function to fit a Causal Forest model

    This function provides a simple interface to fit Causal Forest models,
    similar to the regress() function for OLS.

    Parameters
    ----------
    formula : str, optional
        Formula specification: "Y ~ T | X1 + X2 + ... [| W1 + W2 + ...]"
    data : pd.DataFrame, optional
        Data containing all variables if using formula interface
    Y : array-like, optional
        Outcome variable
    T : array-like, optional
        Treatment variable
    X : array-like, optional
        Effect modifier variables
    W : array-like, optional
        Control variables
    n_estimators : int, default=100
        Number of trees in the forest
    min_samples_leaf : int, default=5
        Minimum samples per leaf
    max_depth : int, optional
        Maximum tree depth
    max_samples : float, default=0.5
        Fraction of samples per tree
    model_y : estimator, optional
        First-stage outcome model
    model_t : estimator, optional
        First-stage treatment model
    discrete_treatment : bool, default=True
        Whether treatment is discrete
    random_state : int, optional
        Random seed
    **kwargs
        Additional arguments passed to CausalForest

    Returns
    -------
    result : CausalForest
        Fitted Causal Forest model

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from statspai.forest import causal_forest
    >>>
    >>> # Generate sample data
    >>> np.random.seed(42)
    >>> n = 500
    >>> X = np.random.normal(0, 1, (n, 2))
    >>> T = np.random.binomial(1, 0.5, n)
    >>> Y = X[:, 0] * T + X[:, 1] + np.random.normal(0, 0.5, n)
    >>> data = pd.DataFrame({
    ...     'outcome': Y, 'treatment': T, 'X1': X[:, 0], 'X2': X[:, 1]
    ... })
    >>>
    >>> # Fit using formula interface
    >>> cf = causal_forest('outcome ~ treatment | X1 + X2', data=data,
    ...                   n_estimators=50, random_state=42)
    >>> print(cf.summary())
    >>>
    >>> # Estimate effects
    >>> effects = cf.effect(data[['X1', 'X2']])
    >>> print(f"Mean effect: {effects.mean():.3f}")
    """
    cf = CausalForest(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        max_samples=max_samples,
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=discrete_treatment,
        random_state=random_state,
        **kwargs,
    )

    cf.fit(formula=formula, data=data, Y=Y, T=T, X=X, W=W)
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            cf,
            function="sp.causal_forest",
            params={
                "formula": formula,
                "n_estimators": n_estimators,
                "min_samples_leaf": min_samples_leaf,
                "max_depth": max_depth,
                "model_y": type(model_y).__name__ if model_y is not None else None,
                "model_t": type(model_t).__name__ if model_t is not None else None,
                "discrete_treatment": discrete_treatment,
                "random_state": random_state,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return cf
