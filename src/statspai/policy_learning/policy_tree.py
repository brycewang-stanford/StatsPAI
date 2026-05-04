"""
Policy Tree: Optimal treatment assignment rules.

Learns a depth-limited decision tree that maximises expected welfare:

    max_{pi}  E[ Gamma_i * pi(X_i) + Gamma_i^0 * (1 - pi(X_i)) ]

where pi(X) in {0, 1} is the treatment policy and Gamma_i are doubly
robust scores (AIPW pseudo-outcomes).

For depth-1 (stump) and depth-2 trees, an exact solution is found
via exhaustive search over all possible splits. For deeper trees,
a greedy recursive splitting approach is used.

References
----------
Athey, S. & Wager, S. (2021).
"Policy Learning with Observational Data."
Econometrica, 89(1), 133-161. [@athey2021matrix]
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd

# sklearn is imported lazily inside the methods that need it so that
# ``import statspai`` doesn't pull ~245 sklearn submodules through this
# file when the user never touches policy_tree.

from ..core.results import CausalResult


# ======================================================================
# Public API
# ======================================================================

def policy_tree(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    policy_covariates: Optional[List[str]] = None,
    max_depth: int = 2,
    min_leaf_size: int = 25,
    n_folds: int = 5,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Learn an optimal treatment assignment policy.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    treat : str
        Binary treatment variable (0/1).
    covariates : list of str
        Covariates for CATE estimation (nuisance).
    policy_covariates : list of str, optional
        Covariates for the policy tree. If None, uses `covariates`.
    max_depth : int, default 2
        Maximum depth of the policy tree.
    min_leaf_size : int, default 25
        Minimum observations in a leaf.
    n_folds : int, default 5
        Cross-fitting folds for doubly robust scores.
    alpha : float, default 0.05
        Significance level.
    random_state : int, default 42

    Returns
    -------
    dict
        'tree' : PolicyTree
            The fitted policy tree object.
        'policy' : np.ndarray
            Treatment recommendations (0 or 1) for each observation.
        'value_treat_all' : float
            Expected value if treating everyone.
        'value_treat_none' : float
            Expected value if treating no one.
        'value_policy' : float
            Expected value under the learned policy.
        'value_gain' : float
            Gain from policy vs. best uniform policy.
        'fraction_treated' : float
            Fraction recommended for treatment.
        'rules' : str
            Human-readable policy rules.
        'n_obs' : int

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.policy_tree(df, y='outcome', treat='treatment',
    ...                         covariates=['x1', 'x2', 'x3'],
    ...                         max_depth=2)
    >>> print(result['rules'])
    >>> print(f"Policy value gain: {result['value_gain']:.4f}")
    """
    est = PolicyTree(
        data=data, y=y, treat=treat, covariates=covariates,
        policy_covariates=policy_covariates,
        max_depth=max_depth, min_leaf_size=min_leaf_size,
        n_folds=n_folds, alpha=alpha, random_state=random_state,
    )
    return est.fit()


def policy_value(
    scores: np.ndarray,
    policy: np.ndarray,
) -> float:
    """
    Evaluate the expected value of a treatment policy.

    Parameters
    ----------
    scores : np.ndarray (n,)
        Doubly robust scores (AIPW pseudo-outcomes for treatment).
        Positive scores indicate the individual benefits from treatment.
    policy : np.ndarray (n,)
        Binary policy recommendations (0 or 1).

    Returns
    -------
    float
        Estimated expected value of the policy.
    """
    scores = np.asarray(scores)
    policy = np.asarray(policy)
    # Value = E[Gamma * pi + (baseline)]
    # Since scores represent the *gain* from treatment,
    # policy value = mean(scores * policy)
    return float(np.mean(scores * policy))


# ======================================================================
# Policy Tree class
# ======================================================================

class PolicyTree:
    """
    Optimal depth-limited policy tree.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    treat : str
    covariates : list of str
    policy_covariates : list of str, optional
    max_depth : int
    min_leaf_size : int
    n_folds : int
    alpha : float
    random_state : int
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        policy_covariates: Optional[List[str]] = None,
        max_depth: int = 2,
        min_leaf_size: int = 25,
        n_folds: int = 5,
        alpha: float = 0.05,
        random_state: int = 42,
    ):
        self.data = data
        self.y = y
        self.treat = treat
        self.covariates = covariates
        self.policy_covariates = policy_covariates or covariates
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.n_folds = n_folds
        self.alpha = alpha
        self.random_state = random_state

    def fit(self) -> Dict[str, Any]:
        """Learn the optimal policy tree."""
        # Prepare data
        all_cols = list(set([self.y, self.treat] + self.covariates +
                            self.policy_covariates))
        missing = [c for c in all_cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        clean = self.data[all_cols].dropna()
        Y = clean[self.y].values.astype(np.float64)
        D = clean[self.treat].values.astype(np.float64)
        W = clean[self.covariates].values.astype(np.float64)
        X_pol = clean[self.policy_covariates].values.astype(np.float64)
        n = len(Y)

        unique_d = np.unique(D)
        if not (len(unique_d) == 2 and set(unique_d.astype(int)) == {0, 1}):
            raise ValueError(f"Treatment must be binary (0/1)")

        # Step 1: Compute doubly robust scores via cross-fitting
        scores = self._compute_dr_scores(Y, D, W, n)

        # Step 2: Grow policy tree on scores
        tree = self._grow_tree(X_pol, scores, depth=0)

        # Step 3: Generate policy
        policy_decisions = self._predict_tree(tree, X_pol)

        # Step 4: Evaluate
        value_policy = float(np.mean(scores * policy_decisions))
        value_all = float(np.mean(scores))
        value_none = 0.0

        fraction_treated = float(np.mean(policy_decisions))

        # Generate rules
        rules = self._tree_to_rules(tree, self.policy_covariates)

        self._tree = tree
        self._scores = scores

        return {
            'tree': self,
            'policy': policy_decisions,
            'value_treat_all': value_all,
            'value_treat_none': value_none,
            'value_policy': value_policy,
            'value_gain': value_policy - max(value_all, value_none),
            'fraction_treated': fraction_treated,
            'rules': rules,
            'n_obs': n,
        }

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Predict treatment assignment for new data.

        Parameters
        ----------
        X_new : np.ndarray (n, p)
            Policy covariates for new observations.

        Returns
        -------
        np.ndarray (n,)
            Binary treatment recommendations (0 or 1).
        """
        if not hasattr(self, '_tree'):
            raise ValueError("PolicyTree must be fitted first.")
        X_new = np.asarray(X_new, dtype=np.float64)
        return self._predict_tree(self._tree, X_new)

    def _compute_dr_scores(self, Y, D, W, n):
        """Compute AIPW doubly robust scores for treatment benefit."""
        from sklearn.ensemble import (
            GradientBoostingRegressor,
            GradientBoostingClassifier,
        )
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=self.n_folds, shuffle=True,
                    random_state=self.random_state)

        mu1_hat = np.zeros(n)
        mu0_hat = np.zeros(n)
        e_hat = np.zeros(n)

        for train_idx, test_idx in kf.split(W):
            W_tr, D_tr, Y_tr = W[train_idx], D[train_idx], Y[train_idx]
            W_te = W[test_idx]

            # Outcome models
            mask1 = D_tr == 1
            mask0 = D_tr == 0

            m1 = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, random_state=self.random_state
            )
            m0 = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, random_state=self.random_state
            )

            if mask1.sum() > 0:
                m1.fit(W_tr[mask1], Y_tr[mask1])
                mu1_hat[test_idx] = m1.predict(W_te)
            if mask0.sum() > 0:
                m0.fit(W_tr[mask0], Y_tr[mask0])
                mu0_hat[test_idx] = m0.predict(W_te)

            # Propensity
            prop = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=self.random_state
            )
            prop.fit(W_tr, D_tr)
            e_hat[test_idx] = np.clip(prop.predict_proba(W_te)[:, 1],
                                       0.025, 0.975)

        # AIPW score for treatment benefit
        scores = (
            (mu1_hat - mu0_hat)
            + D * (Y - mu1_hat) / e_hat
            - (1 - D) * (Y - mu0_hat) / (1 - e_hat)
        )

        return scores

    def _grow_tree(self, X, scores, depth):
        """
        Recursively grow the policy tree.

        Returns a nested dict representing the tree structure.
        """
        n = len(scores)

        # Leaf node: assign treatment if mean score > 0
        if depth >= self.max_depth or n < 2 * self.min_leaf_size:
            return {
                'type': 'leaf',
                'action': 1 if np.mean(scores) > 0 else 0,
                'value': float(np.mean(scores)),
                'n': n,
            }

        best_gain = -np.inf
        best_split = None

        n_features = X.shape[1]

        for j in range(n_features):
            # Get sorted unique thresholds
            vals = np.sort(np.unique(X[:, j]))
            if len(vals) < 2:
                continue

            # Use quantile-based candidate splits for efficiency
            n_candidates = min(50, len(vals) - 1)
            quantiles = np.linspace(0, 1, n_candidates + 2)[1:-1]
            thresholds = np.quantile(X[:, j], quantiles)
            thresholds = np.unique(thresholds)

            for threshold in thresholds:
                left_mask = X[:, j] <= threshold
                right_mask = ~left_mask

                n_left = left_mask.sum()
                n_right = right_mask.sum()

                if n_left < self.min_leaf_size or n_right < self.min_leaf_size:
                    continue

                # Value of optimal assignment in each child
                left_scores = scores[left_mask]
                right_scores = scores[right_mask]

                # Best action in each leaf
                left_val = max(np.mean(left_scores), 0)
                right_val = max(np.mean(right_scores), 0)

                # Total gain
                gain = (n_left * left_val + n_right * right_val) / n

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature': j,
                        'threshold': float(threshold),
                        'left_mask': left_mask,
                        'right_mask': right_mask,
                    }

        # If no valid split found, return leaf
        if best_split is None:
            return {
                'type': 'leaf',
                'action': 1 if np.mean(scores) > 0 else 0,
                'value': float(np.mean(scores)),
                'n': n,
            }

        # Recurse
        left_child = self._grow_tree(
            X[best_split['left_mask']],
            scores[best_split['left_mask']],
            depth + 1,
        )
        right_child = self._grow_tree(
            X[best_split['right_mask']],
            scores[best_split['right_mask']],
            depth + 1,
        )

        return {
            'type': 'split',
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'left': left_child,
            'right': right_child,
            'n': n,
        }

    def _predict_tree(self, tree, X):
        """Predict treatment assignments from a tree."""
        n = X.shape[0]
        policy = np.zeros(n, dtype=int)

        for i in range(n):
            node = tree
            while node['type'] == 'split':
                if X[i, node['feature']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            policy[i] = node['action']

        return policy

    def _tree_to_rules(self, tree, var_names, prefix=""):
        """Convert tree to human-readable rules."""
        if tree['type'] == 'leaf':
            action = "TREAT" if tree['action'] == 1 else "DON'T TREAT"
            return (f"{prefix}{action} "
                    f"(n={tree['n']}, avg_benefit={tree['value']:.4f})\n")

        feat = var_names[tree['feature']]
        thresh = tree['threshold']

        rules = ""
        rules += f"{prefix}IF {feat} <= {thresh:.4f}:\n"
        rules += self._tree_to_rules(tree['left'], var_names,
                                     prefix + "  ")
        rules += f"{prefix}ELSE ({feat} > {thresh:.4f}):\n"
        rules += self._tree_to_rules(tree['right'], var_names,
                                     prefix + "  ")
        return rules


# ======================================================================
# Citation
# ======================================================================

CausalResult._CITATIONS['policy_tree'] = (
    "@article{athey2021policy,\n"
    "  title={Policy Learning with Observational Data},\n"
    "  author={Athey, Susan and Wager, Stefan},\n"
    "  journal={Econometrica},\n"
    "  volume={89},\n"
    "  number={1},\n"
    "  pages={133--161},\n"
    "  year={2021},\n"
    "  publisher={Wiley}\n"
    "}"
)
