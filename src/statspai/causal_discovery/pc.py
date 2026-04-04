"""
PC Algorithm: Constraint-based causal discovery.

The PC algorithm (Spirtes, Glymour, Scheines 2000) learns a CPDAG
(completed partially directed acyclic graph) from observational data
using conditional independence tests.

Steps:
1. Start with a complete undirected graph.
2. For increasing conditioning set size k = 0, 1, 2, ...:
   - For each adjacent pair (X, Y), test X _||_ Y | S for all
     subsets S of size k from adj(X) \\ {Y}.
   - If any test yields independence, remove the edge X—Y and
     record S as the separating set.
3. Orient edges using three rules (v-structures, acyclicity, completeness).

References
----------
Spirtes, P., Glymour, C., & Scheines, R. (2000).
Causation, Prediction, and Search (2nd ed.). MIT Press.

Colombo, D. & Maathuis, M. H. (2014).
Order-independent constraint-based causal structure learning.
JMLR, 15, 3921-3962.
"""

from typing import Optional, List, Dict, Any, Set, Tuple
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ======================================================================
# Public API
# ======================================================================

def pc_algorithm(
    data: pd.DataFrame,
    variables: Optional[List[str]] = None,
    alpha: float = 0.05,
    max_cond_size: Optional[int] = None,
    ci_test: str = 'fisherz',
) -> Dict[str, Any]:
    """
    Learn causal structure using the PC algorithm.

    Parameters
    ----------
    data : pd.DataFrame
        Observational data (n_samples x d_variables).
    variables : list of str, optional
        Column names to use. If None, uses all numeric columns.
    alpha : float, default 0.05
        Significance level for conditional independence tests.
        Lower alpha = sparser graph (fewer edges).
    max_cond_size : int, optional
        Maximum conditioning set size. If None, goes up to d-2.
    ci_test : str, default 'fisherz'
        Conditional independence test: 'fisherz' (partial correlation)
        or 'hsic' (kernel-based, non-linear).

    Returns
    -------
    dict
        'skeleton' : pd.DataFrame
            Undirected adjacency matrix (0/1).
        'cpdag' : pd.DataFrame
            CPDAG adjacency matrix. cpdag[i,j] = 1 means i -> j.
            If both cpdag[i,j] = 1 and cpdag[j,i] = 1, the edge
            is undirected (i -- j).
        'edges' : list of tuples
            Directed edges as (parent, child) tuples.
        'undirected_edges' : list of tuples
            Undirected edges as (node1, node2) tuples.
        'separating_sets' : dict
            {(i, j): set} of separating sets for removed edges.
        'variables' : list of str
        'n_edges' : int
        'n_obs' : int
        'alpha' : float
        'ci_test' : str

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.pc_algorithm(df, variables=['X', 'Z', 'M', 'Y'])
    >>> print(result['edges'])       # directed edges
    >>> print(result['cpdag'])       # CPDAG adjacency matrix
    """
    est = PCAlgorithm(
        data=data, variables=variables, alpha=alpha,
        max_cond_size=max_cond_size, ci_test=ci_test,
    )
    return est.fit()


# ======================================================================
# PC Algorithm Estimator
# ======================================================================

class PCAlgorithm:
    """
    PC Algorithm for causal discovery.

    Parameters
    ----------
    data : pd.DataFrame
    variables : list of str, optional
    alpha : float
    max_cond_size : int, optional
    ci_test : str
    """

    def __init__(
        self,
        data: pd.DataFrame,
        variables: Optional[List[str]] = None,
        alpha: float = 0.05,
        max_cond_size: Optional[int] = None,
        ci_test: str = 'fisherz',
    ):
        self.data = data
        self.variables = variables
        self.alpha = alpha
        self.max_cond_size = max_cond_size
        self.ci_test = ci_test

    def fit(self) -> Dict[str, Any]:
        """Run the PC algorithm and return learned structure."""
        # Prepare data
        if self.variables is not None:
            missing = [v for v in self.variables if v not in self.data.columns]
            if missing:
                raise ValueError(f"Variables not found in data: {missing}")
            X = self.data[self.variables].dropna().values.astype(np.float64)
            var_names = list(self.variables)
        else:
            numeric = self.data.select_dtypes(include=[np.number])
            X = numeric.dropna().values.astype(np.float64)
            var_names = list(numeric.columns)

        n, d = X.shape
        if d < 2:
            raise ValueError("At least 2 variables are required")

        max_k = self.max_cond_size if self.max_cond_size is not None else d - 2

        # Step 1: Learn skeleton
        adj, sep_sets = self._learn_skeleton(X, d, n, max_k)

        # Save skeleton
        skeleton = adj.copy()

        # Step 2: Orient edges (v-structures + Meek rules)
        cpdag = self._orient_edges(adj, sep_sets, d)

        # Build edge lists
        directed_edges = []
        undirected_edges = set()
        for i in range(d):
            for j in range(d):
                if cpdag[i, j] == 1:
                    if cpdag[j, i] == 1:
                        # Undirected: add once
                        pair = (min(i, j), max(i, j))
                        undirected_edges.add(pair)
                    else:
                        directed_edges.append((var_names[i], var_names[j]))

        undirected_edge_names = [(var_names[i], var_names[j])
                                 for i, j in undirected_edges]

        # Convert sep_sets to use variable names
        sep_sets_named = {}
        for (i, j), s in sep_sets.items():
            key = (var_names[i], var_names[j])
            sep_sets_named[key] = {var_names[k] for k in s}

        skeleton_df = pd.DataFrame(skeleton, index=var_names, columns=var_names)
        cpdag_df = pd.DataFrame(cpdag, index=var_names, columns=var_names)

        total_edges = len(directed_edges) + len(undirected_edge_names)

        self._cpdag = cpdag
        self._var_names = var_names

        return {
            'skeleton': skeleton_df,
            'cpdag': cpdag_df,
            'edges': directed_edges,
            'undirected_edges': undirected_edge_names,
            'separating_sets': sep_sets_named,
            'variables': var_names,
            'n_edges': total_edges,
            'n_obs': n,
            'alpha': self.alpha,
            'ci_test': self.ci_test,
        }

    def _learn_skeleton(self, X, d, n, max_k):
        """
        Phase I: Learn the skeleton via conditional independence tests.

        Start with complete graph, remove edges where CI holds.
        """
        # Adjacency matrix (symmetric, 1 = edge)
        adj = np.ones((d, d), dtype=int)
        np.fill_diagonal(adj, 0)

        # Separating sets
        sep_sets: Dict[Tuple[int, int], set] = {}

        for k in range(max_k + 1):
            # For each pair of adjacent nodes
            edges_to_test = []
            for i in range(d):
                for j in range(i + 1, d):
                    if adj[i, j] == 1:
                        edges_to_test.append((i, j))

            for i, j in edges_to_test:
                if adj[i, j] == 0:
                    continue  # already removed

                # Neighbours of i excluding j
                nbrs_i = set(np.where(adj[i] == 1)[0]) - {j}
                # Neighbours of j excluding i
                nbrs_j = set(np.where(adj[j] == 1)[0]) - {i}

                # Union of neighbours (order-independent variant)
                candidates = nbrs_i | nbrs_j

                if len(candidates) < k:
                    continue

                # Test all subsets of size k
                found_independent = False
                for S in combinations(candidates, k):
                    S_set = set(S)
                    pval = self._ci_test_pval(X, i, j, list(S_set), n)

                    if pval > self.alpha:
                        # Independent: remove edge
                        adj[i, j] = 0
                        adj[j, i] = 0
                        sep_sets[(i, j)] = S_set
                        sep_sets[(j, i)] = S_set
                        found_independent = True
                        break

                if found_independent:
                    continue

        return adj, sep_sets

    def _orient_edges(self, adj, sep_sets, d):
        """
        Phase II: Orient edges to form CPDAG.

        1. Orient v-structures: X -> Z <- Y if X-Z-Y and Z not in sep(X,Y).
        2. Apply Meek's rules for completeness.
        """
        # Start with the skeleton as a directed graph
        # cpdag[i,j] = 1 means i -> j (or i -- j if cpdag[j,i] also 1)
        cpdag = adj.copy()

        # Rule 1: Orient v-structures (colliders)
        for j in range(d):
            # Find all pairs of non-adjacent nodes connected through j
            nbrs = list(np.where(adj[j] == 1)[0])
            for idx_a in range(len(nbrs)):
                for idx_b in range(idx_a + 1, len(nbrs)):
                    i = nbrs[idx_a]
                    k = nbrs[idx_b]

                    # i and k must be non-adjacent
                    if adj[i, k] == 1:
                        continue

                    # Check if j is NOT in the separating set of (i, k)
                    sep_key = (min(i, k), max(i, k))
                    if sep_key in sep_sets:
                        if j not in sep_sets[sep_key]:
                            # Orient as i -> j <- k (v-structure)
                            cpdag[j, i] = 0  # remove j -> i
                            cpdag[j, k] = 0  # remove j -> k
                            # Keep i -> j and k -> j

        # Meek's rules (iterate until no changes)
        changed = True
        while changed:
            changed = False

            for i in range(d):
                for j in range(d):
                    if cpdag[i, j] == 0 or i == j:
                        continue

                    # Rule 2: If i -> j -- k and i and k are not adjacent,
                    # orient j -> k
                    if cpdag[j, i] == 0:  # i -> j (directed)
                        for k in range(d):
                            if k == i or k == j:
                                continue
                            if cpdag[j, k] == 1 and cpdag[k, j] == 1:
                                # j -- k (undirected)
                                if cpdag[i, k] == 0 and cpdag[k, i] == 0:
                                    # i and k not adjacent
                                    cpdag[k, j] = 0  # orient j -> k
                                    changed = True

                    # Rule 3: If i -- j and there exists k such that
                    # i -> k -> j, orient i -> j
                    if cpdag[i, j] == 1 and cpdag[j, i] == 1:
                        # i -- j undirected
                        for k in range(d):
                            if k == i or k == j:
                                continue
                            if (cpdag[i, k] == 1 and cpdag[k, i] == 0
                                    and cpdag[k, j] == 1 and cpdag[j, k] == 0):
                                # i -> k -> j
                                cpdag[j, i] = 0  # orient i -> j
                                changed = True
                                break

        return cpdag

    def _ci_test_pval(self, X, i, j, S, n):
        """
        Conditional independence test: X_i _||_ X_j | X_S.

        Returns p-value.
        """
        if self.ci_test == 'fisherz':
            return _fisher_z_test(X, i, j, S, n)
        else:
            raise ValueError(f"Unknown CI test: {self.ci_test}. Use 'fisherz'.")

    def summary(self) -> str:
        """Print a summary of the learned structure."""
        if not hasattr(self, '_cpdag'):
            raise ValueError("Model must be fitted first. Call .fit()")

        d = len(self._var_names)
        lines = []
        lines.append("=" * 60)
        lines.append("  PC Algorithm: Causal Discovery")
        lines.append("  Spirtes, Glymour, Scheines (2000)")
        lines.append("=" * 60)
        lines.append(f"  Variables: {', '.join(self._var_names)}")
        lines.append(f"  Alpha: {self.alpha}")
        lines.append(f"  CI Test: {self.ci_test}")
        lines.append("")

        directed = []
        undirected = set()
        for i in range(d):
            for j in range(d):
                if self._cpdag[i, j] == 1:
                    if self._cpdag[j, i] == 1:
                        pair = (min(i, j), max(i, j))
                        undirected.add(pair)
                    else:
                        directed.append((i, j))

        if directed:
            lines.append("  Directed Edges:")
            lines.append("  " + "-" * 40)
            for i, j in directed:
                lines.append(f"    {self._var_names[i]} -> {self._var_names[j]}")

        if undirected:
            lines.append("  Undirected Edges:")
            lines.append("  " + "-" * 40)
            for i, j in undirected:
                lines.append(f"    {self._var_names[i]} -- {self._var_names[j]}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ======================================================================
# Conditional Independence Tests
# ======================================================================

def _fisher_z_test(X, i, j, S, n):
    """
    Fisher's Z test for conditional independence via partial correlation.

    Tests H0: rho(X_i, X_j | X_S) = 0 using Fisher's Z transformation.

    Parameters
    ----------
    X : np.ndarray (n, d)
    i, j : int
        Variable indices to test.
    S : list of int
        Conditioning set indices.
    n : int
        Sample size.

    Returns
    -------
    float
        Two-sided p-value.
    """
    if len(S) == 0:
        # Marginal correlation
        r = np.corrcoef(X[:, i], X[:, j])[0, 1]
    else:
        # Partial correlation via regression residuals
        r = _partial_correlation(X, i, j, S)

    # Clip for numerical stability
    r = np.clip(r, -1 + 1e-10, 1 - 1e-10)

    # Fisher's Z transformation
    z = 0.5 * np.log((1 + r) / (1 - r))
    # Under H0, sqrt(n - |S| - 3) * z ~ N(0, 1)
    dof = n - len(S) - 3
    if dof < 1:
        return 1.0  # not enough degrees of freedom

    z_stat = np.sqrt(dof) * abs(z)
    pval = 2 * (1 - sp_stats.norm.cdf(z_stat))
    return float(pval)


def _partial_correlation(X, i, j, S):
    """
    Compute partial correlation of X_i and X_j given X_S.

    Uses the formula via the inverse of the sub-covariance matrix.
    """
    idx = [i, j] + list(S)
    sub = X[:, idx]
    C = np.corrcoef(sub, rowvar=False)

    try:
        P = np.linalg.inv(C)
        # Partial correlation = -P[0,1] / sqrt(P[0,0] * P[1,1])
        denom = np.sqrt(abs(P[0, 0] * P[1, 1]))
        if denom < 1e-15:
            return 0.0
        return -P[0, 1] / denom
    except np.linalg.LinAlgError:
        return 0.0
