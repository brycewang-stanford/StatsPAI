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
from typing import Optional, Union, Tuple, Dict, Any, List
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeRegressor
import warnings

# Import our core classes
from ..core.base import BaseModel
from ..core.results import EconometricResults
from ..core.utils import parse_formula


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
    >>> from statspai.causal import CausalForest
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
        self.model_y = model_y
        self.model_t = model_t
        self.discrete_treatment = discrete_treatment
        self.honest = honest
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Initialize default models if not provided
        if self.model_y is None:
            self.model_y = RandomForestRegressor(
                n_estimators=100, random_state=random_state
            )
        if self.model_t is None:
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
        self._forest = None
        self._treatment_values = None
        self._feature_names = None
        
    def fit(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        Y: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        W: Optional[np.ndarray] = None,
    ) -> 'CausalForest':
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
        # Parse inputs
        if formula is not None and data is not None:
            Y, T, X, W = self._parse_formula_inputs(formula, data)
        elif Y is not None and T is not None and X is not None:
            Y, T, X, W = self._validate_array_inputs(Y, T, X, W)
        else:
            raise ValueError(
                "Must provide either (formula, data) or (Y, T, X) arguments"
            )
        
        # Validate inputs
        Y = np.asarray(Y).ravel()
        T = np.asarray(T).ravel()
        X = np.asarray(X)
        if W is not None:
            W = np.asarray(W)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if W is not None and W.ndim == 1:
            W = W.reshape(-1, 1)
            
        n_samples = len(Y)
        if len(T) != n_samples or len(X) != n_samples:
            raise ValueError("Y, T, and X must have the same number of samples")
        if W is not None and len(W) != n_samples:
            raise ValueError("W must have the same number of samples as Y")
        
        # Store data info
        self.data_info = {
            'nobs': n_samples,
            'n_features': X.shape[1],
            'n_controls': W.shape[1] if W is not None else 0,
            'treatment_values': np.unique(T),
        }
        
        self._treatment_values = np.unique(T)
        self._feature_names = [f'X{i}' for i in range(X.shape[1])]
        self._X_original = X.copy()
        self._T_original = T.copy()
        self._Y_original = Y.copy()
        
        # Validate treatment
        if self.discrete_treatment:
            if len(self._treatment_values) < 2:
                raise ValueError("Need at least 2 treatment values for discrete treatment")
        
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
                self.model_t, first_stage_features, T, cv=3, 
                method='predict_proba', n_jobs=self.n_jobs
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
            'method': 'Causal Forest',
            'n_estimators': self.n_estimators,
            'n_features': X.shape[1],
            'average_treatment_effect': ate,
            'treatment_type': 'discrete' if self.discrete_treatment else 'continuous',
        }
        
        return self
    
    def _parse_formula_inputs(
        self, formula: str, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Parse formula inputs to extract Y, T, X, W arrays"""
        # Parse formula: "Y ~ T | X1 + X2 + ... [| W1 + W2 + ...]"
        parts = formula.split('|')
        if len(parts) < 2:
            raise ValueError(
                "Formula must have format 'Y ~ T | X1 + X2 + ...' or "
                "'Y ~ T | X1 + X2 + ... | W1 + W2 + ...'"
            )
        
        # Parse outcome and treatment
        yt_part = parts[0].strip()
        if '~' not in yt_part:
            raise ValueError("Formula must contain '~' to separate outcome and treatment")
        
        y_name, t_name = yt_part.split('~')
        y_name = y_name.strip()
        t_name = t_name.strip()
        
        # Parse effect modifiers (X)
        x_part = parts[1].strip()
        x_names = [name.strip() for name in x_part.split('+')]
        
        # Parse controls (W) if provided
        w_names = []
        if len(parts) > 2:
            w_part = parts[2].strip()
            w_names = [name.strip() for name in w_part.split('+')]
        
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
                max_features='sqrt' if n_features > 1 else None,
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
                    tree, X[estimate_indices], T_residual[estimate_indices], 
                    Y_residual[estimate_indices]
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
    ):
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
                        # Simple ratio estimator: Cov(Y,T) / Var(T)
                        causal_effect = np.cov(y_leaf, t_leaf)[0, 1] / np.var(t_leaf)
                    else:
                        causal_effect = 0.0
                else:
                    # Multi-class treatment - use first treatment effect
                    if np.var(t_leaf[:, 0]) > 1e-8:
                        causal_effect = np.cov(y_leaf, t_leaf[:, 0])[0, 1] / np.var(t_leaf[:, 0])
                    else:
                        causal_effect = 0.0
            else:
                # Continuous treatment
                if np.var(t_leaf) > 1e-8:
                    causal_effect = np.cov(y_leaf, t_leaf)[0, 1] / np.var(t_leaf)
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
            raise ValueError("Model must be fitted before predicting effects")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Average predictions across all trees
        predictions = []
        for tree in self._forest:
            pred = tree.predict(X)
            # Ensure pred is 1D for each tree
            if pred.ndim > 1:
                pred = pred.ravel()
            predictions.append(pred)
        
        # Return average effect across trees
        predictions = np.array(predictions)  # shape: (n_trees, n_samples)
        return np.mean(predictions, axis=0)  # shape: (n_samples,)
    
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
            raise ValueError("Model must be fitted before making predictions")
        
        if data is None:
            if hasattr(self, '_X_original'):
                X = self._X_original
            else:
                raise ValueError("No data provided and no training data available")
        else:
            if isinstance(data, pd.DataFrame):
                if hasattr(self, '_feature_names'):
                    X = data[self._feature_names].values
                else:
                    # Use all numeric columns
                    X = data.select_dtypes(include=[np.number]).values
            else:
                X = np.asarray(data)
        
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
            raise ValueError("Model must be fitted before computing intervals")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Use bootstrap of little bags approach
        # Collect predictions from each tree (which are already bootstrap samples)
        predictions = []
        for tree in self._forest:
            pred = tree.predict(X)
            # Ensure pred is 1D
            if pred.ndim > 1:
                pred = pred.ravel()
            predictions.append(pred)
        
        # Convert to array and ensure consistent shapes
        try:
            predictions = np.array(predictions)  # shape: (n_trees, n_samples)
        except ValueError:
            # Handle inconsistent shapes by ensuring all predictions have same length
            min_len = min(len(p) for p in predictions)
            predictions = np.array([p[:min_len] for p in predictions])
        
        # Compute percentiles across trees for each sample
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        lower = np.percentile(predictions, lower_percentile, axis=0)
        upper = np.percentile(predictions, upper_percentile, axis=0)
        
        return lower, upper
    
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
        
        ate = self.diagnostics.get('average_treatment_effect', 0)
        
        summary_lines = [
            "=" * 60,
            "Causal Forest Results",
            "=" * 60,
            f"Method:                   Causal Forest",
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
            raise ValueError("Model must be fitted before computing importance")
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
    ) -> pd.DataFrame:
        """Best Linear Projection (BLP) heterogeneity test (Chernozhukov et al. 2020).

        Regress CATE(X_i) on the features X_i:

            CATE_i = β₀ + X_i' β₁ + ε_i

        A joint F-test on β₁ = 0 tests for systematic treatment-effect
        heterogeneity. Individual t-tests indicate which features drive it.

        Returns a DataFrame with coefficient, SE, t-stat, p-value per feature.
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted first")
        X = X_test if X_test is not None else self._X_original
        X = np.asarray(X)
        cate = self.effect(X)
        n, k = X.shape
        # Standardise X so coefficients are comparable
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
        D = np.column_stack([np.ones(n), X_std])
        DtD_inv = np.linalg.inv(D.T @ D)
        beta = DtD_inv @ (D.T @ cate)
        e = cate - D @ beta
        sigma2 = float(e @ e) / (n - k - 1)
        se = np.sqrt(np.diag(sigma2 * DtD_inv))
        tvals = beta / se
        from scipy import stats
        pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), df=n - k - 1))
        names = ["Intercept"] + (
            self._feature_names or [f"X{j}" for j in range(k)]
        )
        return pd.DataFrame({
            "coef": beta, "se": se, "t": tvals, "p": pvals,
        }, index=names)

    def ate(self, X: Optional[np.ndarray] = None) -> float:
        """Average Treatment Effect (mean CATE)."""
        return float(self.effect(X if X is not None else self._X_original).mean())

    def att(self, X: Optional[np.ndarray] = None,
            T: Optional[np.ndarray] = None) -> float:
        """Average Treatment Effect on the Treated."""
        if T is None:
            if hasattr(self, "_T_original"):
                T = self._T_original
            else:
                raise ValueError("Treatment vector needed for ATT")
        X = X if X is not None else self._X_original
        cate = self.effect(X)
        mask = np.asarray(T).ravel() == 1
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
    **kwargs
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
    >>> from statspai.causal import causal_forest
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
        **kwargs
    )
    
    cf.fit(formula=formula, data=data, Y=Y, T=T, X=X, W=W)
    return cf
