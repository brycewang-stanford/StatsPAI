"""
PC Algorithm: Constraint-based causal discovery.

The PC algorithm (Spirtes, Glymour, Scheines 2000) learns a CPDAG
(completed partially directed acyclic graph) from observational data
using conditional independence tests.

Steps:
1. Start with a complete undirected graph.
2. For increasing conditioning set size k = 0, 1, 2, ...:
   - For each adjacent pair (X, Y), test X _||_ Y | S for all
     subsets S of size k from adj(X) \\ {Y}, then from adj(Y) \\ {X},
     with the neighbourhoods fixed at the start of the level (PC-stable).
   - If any test yields independence, remove the edge X—Y and
     record S as the separating set.
3. Orient edges using three rules (v-structures, acyclicity, completeness).

References
----------
Spirtes, P., Glymour, C., & Scheines, R. (2000).
Causation, Prediction, and Search (2nd ed.). MIT Press. [@spirtes2000causation]

Colombo, D. & Maathuis, M. H. (2014).
Order-independent constraint-based causal structure learning.
JMLR, 15, 3921-3962. [@colombo2014order]
"""

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..dag._ci_tests import discrete_ci_test
from ..exceptions import MethodIncompatibility

#: Tests that cross-tabulate their columns.
_DISCRETE_TESTS = ("chi-square", "g-test")

# ======================================================================
# Public API
# ======================================================================


def pc_algorithm(
    data: pd.DataFrame,
    variables: Optional[List[str]] = None,
    alpha: float = 0.05,
    max_cond_size: Optional[int] = None,
    ci_test: str = "fisherz",
    forbidden: Optional[List[Tuple[str, str]]] = None,
    required: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Learn causal structure using the PC algorithm.

    Parameters
    ----------
    data : pd.DataFrame
        Observational data (n_samples x d_variables).
    variables : list of str, optional
        Column names to use. If None: all numeric columns under
        ``ci_test='fisherz'``, all columns under the categorical tests.
    alpha : float, default 0.05
        Significance level for conditional independence tests.
        Lower alpha = sparser graph (fewer edges).
    max_cond_size : int, optional
        Maximum conditioning set size. If None, goes up to d-2.
    ci_test : {'fisherz', 'chi-square', 'g-test'}, default 'fisherz'
        Conditional independence test.

        - ``'fisherz'``: partial correlation, for numeric columns with
          linear relations.
        - ``'chi-square'`` / ``'g-test'``: for categorical columns.
          Pearson's statistic (or the likelihood ratio) is summed over
          the strata of the conditioning set, with degrees of freedom
          counted from the levels present in each stratum
          (``bnlearn``'s ``"x2-adf"`` / ``"mi-adf"``). An independence
          no stratum can test is not rejected, so with little data per
          stratum edges are removed for lack of evidence: keep
          ``max_cond_size`` small on small samples.
    forbidden : list of (str, str), optional
        Background knowledge: edges that must NOT appear in the final
        graph (treated as undirected — both ``(a, b)`` and ``(b, a)`` are
        forbidden when either is given). The skeleton phase keeps these
        absent regardless of CI test outcomes.
    required : list of (str, str), optional
        Background knowledge: directed edges ``a -> b`` that must appear
        in the CPDAG. The skeleton phase preserves them regardless of CI
        rejection, and the orientation phase pins their direction.

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
        'orientation_conflicts' : list of tuples
            Colliders ``(a, b, c)``, meaning ``a -> b <- c``, that the
            tests imply but that could not be drawn because an earlier
            collider had already oriented one of the two edges the other
            way. Empty when the tests are mutually consistent; anything
            else says the sample contradicts itself about those edges.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> X = rng.normal(size=n)
    >>> Z = 0.8 * X + rng.normal(size=n) * 0.5
    >>> M = 0.7 * Z + rng.normal(size=n) * 0.5
    >>> Y = 0.6 * M + rng.normal(size=n) * 0.5
    >>> df = pd.DataFrame({'X': X, 'Z': Z, 'M': M, 'Y': Y})
    >>> result = sp.pc_algorithm(df, variables=['X', 'Z', 'M', 'Y'])
    >>> bool(result['n_edges'] >= 0)  # CPDAG edge count
    True
    """
    est = PCAlgorithm(
        data=data,
        variables=variables,
        alpha=alpha,
        max_cond_size=max_cond_size,
        ci_test=ci_test,
        forbidden=forbidden,
        required=required,
    )
    return est.fit()


def stable_skeleton(
    adj: np.ndarray,
    sep_sets: Dict[Tuple[int, int], set],
    max_k: int,
    alpha: float,
    pvalue: Any,
    required: Any = frozenset(),
) -> None:
    """Remove from ``adj`` every edge some conditional independence explains.

    PC-stable [@colombo2014order]: within a level the
    neighbourhoods are those at its start, so the skeleton does not depend
    on the order of the variables. Each edge x - y is tested given subsets
    of adj(x) minus y and, separately, of adj(y) minus x: a separating
    set, if one exists, lies inside one of the two. Ordered pairs run by x
    then y and subsets in lexicographic order, as in ``pcalg::skeleton``,
    so the separating sets recorded are the same ones. The search goes on
    until no node has enough neighbours for the next level, not until a
    level happens to remove nothing.

    ``adj`` and ``sep_sets`` are updated in place. ``pvalue(x, y, S)``
    returns the p-value of ``x _||_ y | S``; an edge in ``required`` is
    never removed.
    """
    d = adj.shape[0]
    for k in range(max_k + 1):
        frozen = adj.copy()
        if not any(
            frozen[x, y] and frozen[x].sum() - 1 >= k
            for x in range(d)
            for y in range(d)
        ):
            break
        for x in range(d):
            for y in range(d):
                if x == y or adj[x, y] == 0 or frozen[x, y] == 0:
                    continue
                if (x, y) in required:
                    continue
                nbrs = [int(v) for v in np.flatnonzero(frozen[x]) if v != y]
                if len(nbrs) < k:
                    continue
                for S in combinations(nbrs, k):
                    if pvalue(x, y, list(S)) > alpha:
                        adj[x, y] = 0
                        adj[y, x] = 0
                        sep_sets[(x, y)] = set(S)
                        sep_sets[(y, x)] = set(S)
                        break


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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> from statspai.causal_discovery.pc import PCAlgorithm
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> X = rng.normal(size=n)
    >>> Z = 0.8 * X + rng.normal(size=n) * 0.5
    >>> M = 0.7 * Z + rng.normal(size=n) * 0.5
    >>> Y = 0.6 * M + rng.normal(size=n) * 0.5
    >>> df = pd.DataFrame({'X': X, 'Z': Z, 'M': M, 'Y': Y})
    >>> est = PCAlgorithm(data=df, variables=['X', 'Z', 'M', 'Y'])
    >>> result = est.fit()
    >>> bool(result['n_edges'] >= 0)
    True

    References
    ----------
    [@spirtes2000causation]
    """

    def __init__(
        self,
        data: pd.DataFrame,
        variables: Optional[List[str]] = None,
        alpha: float = 0.05,
        max_cond_size: Optional[int] = None,
        ci_test: str = "fisherz",
        forbidden: Optional[List[Tuple[str, str]]] = None,
        required: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.data = data
        self.variables = variables
        self.alpha = alpha
        self.max_cond_size = max_cond_size
        self.ci_test = ci_test
        # Background knowledge — pairs are stored as plain tuples and
        # resolved to indices once ``var_names`` is known in fit().
        self.forbidden = list(forbidden or [])
        self.required = list(required or [])

    def fit(self) -> Dict[str, Any]:
        """Run the PC algorithm and return learned structure."""
        # Prepare data
        if self.variables is not None:
            missing = [v for v in self.variables if v not in self.data.columns]
            if missing:
                raise ValueError(f"Variables not found in data: {missing}")
            frame = self.data[self.variables].dropna()
        elif self.ci_test in _DISCRETE_TESTS:
            frame = self.data.dropna()
        else:
            frame = self.data.select_dtypes(include=[np.number]).dropna()
        var_names = list(frame.columns)

        if self.ci_test not in ("fisherz",) + _DISCRETE_TESTS:
            raise MethodIncompatibility(
                f"pc_algorithm: ci_test={self.ci_test!r} is not one of "
                f"{('fisherz',) + _DISCRETE_TESTS}."
            )
        if self.ci_test in _DISCRETE_TESTS:
            wide = [c for c in var_names if frame[c].nunique() > 20]
            if wide:
                raise MethodIncompatibility(
                    f"pc_algorithm: ci_test={self.ci_test!r} cross-tabulates "
                    f"the columns, and {wide} take more than 20 distinct "
                    "values.",
                    recovery_hint=(
                        "Bin them first (pd.qcut), drop them with "
                        "variables=[...], or use ci_test='fisherz' on "
                        "numeric data."
                    ),
                    diagnostics={"continuous_columns": wide},
                )
            X = np.empty((len(frame), len(var_names)), dtype=np.int64)
            for k, c in enumerate(var_names):
                X[:, k] = pd.factorize(frame[c], sort=True)[0]
        else:
            labelled = [
                c for c in var_names if not pd.api.types.is_numeric_dtype(frame[c])
            ]
            if labelled:
                raise MethodIncompatibility(
                    "pc_algorithm: ci_test='fisherz' is a partial correlation "
                    f"and needs numeric columns; {labelled} are not.",
                    recovery_hint="Use ci_test='chi-square' for categorical data.",
                    diagnostics={"non_numeric_columns": labelled},
                )
            X = frame.values.astype(np.float64)

        n, d = X.shape
        if d < 2:
            dropped = [c for c in self.data.columns if c not in var_names]
            raise MethodIncompatibility(
                "pc_algorithm: At least 2 variables are required; "
                f"{d} numeric column(s) found"
                + (f" ({len(dropped)} non-numeric left out)" if dropped else "")
                + ".",
                recovery_hint=(
                    "For categorical columns use ci_test='chi-square'."
                    if dropped
                    else "Pass a data frame with at least two columns."
                ),
                diagnostics={"non_numeric_columns": dropped},
            )

        max_k = self.max_cond_size if self.max_cond_size is not None else d - 2

        # Resolve background-knowledge name pairs to integer index pairs.
        # Unknown names are silently ignored — callers may pass edges
        # over a superset (e.g. an LLM-proposed edge involving a column
        # the user excluded) and shouldn't get a hard error.
        name_to_idx = {n: i for i, n in enumerate(var_names)}
        forbidden_idx: set[tuple[int, int]] = set()
        for a, b in self.forbidden:
            ia = name_to_idx.get(a)
            ib = name_to_idx.get(b)
            if ia is None or ib is None or ia == ib:
                continue
            forbidden_idx.add((ia, ib))
            forbidden_idx.add((ib, ia))
        required_idx_directed: list[tuple[int, int]] = []
        required_idx_undirected: set[tuple[int, int]] = set()
        for a, b in self.required:
            ia = name_to_idx.get(a)
            ib = name_to_idx.get(b)
            if ia is None or ib is None or ia == ib:
                continue
            required_idx_directed.append((ia, ib))
            required_idx_undirected.add((ia, ib))
            required_idx_undirected.add((ib, ia))
        # Required edges must not also be forbidden — required wins.
        forbidden_idx -= required_idx_undirected
        self._forbidden_idx = forbidden_idx
        self._required_idx_directed = required_idx_directed
        self._required_idx_undirected = required_idx_undirected

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

        undirected_edge_names = [
            (var_names[i], var_names[j]) for i, j in undirected_edges
        ]

        # Convert sep_sets to use variable names
        sep_sets_named = {}
        for (i, j), s in sep_sets.items():
            key = (var_names[i], var_names[j])
            sep_sets_named[key] = {var_names[k] for k in s}

        skeleton_df = pd.DataFrame(
            skeleton,
            index=var_names,
            columns=var_names,
        )
        cpdag_df = pd.DataFrame(cpdag, index=var_names, columns=var_names)

        total_edges = len(directed_edges) + len(undirected_edge_names)

        self._cpdag = cpdag
        self._var_names = var_names

        from ._viz import DAGDict

        return DAGDict(
            {
                "skeleton": skeleton_df,
                "cpdag": cpdag_df,
                "edges": directed_edges,
                "undirected_edges": undirected_edge_names,
                "separating_sets": sep_sets_named,
                "variables": var_names,
                "n_edges": total_edges,
                "n_obs": n,
                "alpha": self.alpha,
                "ci_test": self.ci_test,
                "orientation_conflicts": [
                    (var_names[a], var_names[b], var_names[c])
                    for a, b, c in getattr(self, "_conflicts", [])
                ],
            }
        )

    def _learn_skeleton(
        self,
        X: np.ndarray,
        d: int,
        n: int,
        max_k: int,
    ) -> tuple[np.ndarray, dict[tuple[int, int], set[int]]]:
        """
        Phase I: Learn the skeleton via conditional independence tests.

        Start with complete graph, remove edges where CI holds.
        """
        # Adjacency matrix (symmetric, 1 = edge)
        adj = np.ones((d, d), dtype=int)
        np.fill_diagonal(adj, 0)

        # Apply background-knowledge edge prohibitions up-front.
        for ia, ib in getattr(self, "_forbidden_idx", set()):
            adj[ia, ib] = 0
            adj[ib, ia] = 0

        # Separating sets
        sep_sets: dict[tuple[int, int], set[int]] = {}

        # Through 1.38.0 the conditioning sets were drawn from the *union* of
        # the two neighbourhoods, which were updated as edges fell. That ran
        # tests no version of PC runs and made the result depend on column
        # order.
        required: set[tuple[int, int]] = getattr(
            self, "_required_idx_undirected", set()
        )
        stable_skeleton(
            adj,
            sep_sets,
            max_k,
            self.alpha,
            lambda x, y, S: self._ci_test_pval(X, x, y, S, n),
            required,
        )

        return adj, sep_sets

    def _orient_edges(
        self,
        adj: np.ndarray,
        sep_sets: dict[tuple[int, int], set[int]],
        d: int,
    ) -> np.ndarray:
        """
        Phase II: Orient edges to form CPDAG.

        1. Orient v-structures: X -> Z <- Y if X-Z-Y and Z not in sep(X,Y).
        2. Apply Meek's rules for completeness.
        """
        # Start with the skeleton as a directed graph
        # cpdag[i,j] = 1 means i -> j (or i -- j if cpdag[j,i] also 1)
        cpdag = adj.copy()

        # Pin background-knowledge directed edges before v-structure
        # orientation. Required edges become a -> b only (forbidden
        # entries on the same pair already had skeleton-level removal
        # rejected upstream because required wins).
        for ia, ib in getattr(self, "_required_idx_directed", []):
            cpdag[ia, ib] = 1
            cpdag[ib, ia] = 0

        # Rule 1: Orient v-structures (colliders).
        #
        # In a finite sample two colliders can claim one edge in opposite
        # directions (A -> T <- R and E -> R <- T both use R - T). Zeroing
        # the reverse entry for each in turn zeroed both, and the edge left
        # the graph: the skeleton had it, the CPDAG did not. An arrowhead
        # is now placed only on an edge that is still undirected or already
        # points that way; the first collider in node order keeps the edge
        # and the clash is recorded.
        conflicts: list[tuple[int, int, int]] = []

        def can_point(a: int, b: int) -> bool:
            """Whether a -> b is compatible with what is already oriented."""
            return bool(cpdag[a, b] == 1)

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
                            # Orient as i -> j <- k (v-structure), unless
                            # an earlier collider has already pointed one
                            # of the two edges away from j.
                            if can_point(i, j) and can_point(k, j):
                                cpdag[j, i] = 0  # remove j -> i
                                cpdag[j, k] = 0  # remove j -> k
                            else:
                                conflicts.append((int(i), int(j), int(k)))
        self._conflicts = conflicts

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
                            if (
                                cpdag[i, k] == 1
                                and cpdag[k, i] == 0
                                and cpdag[k, j] == 1
                                and cpdag[j, k] == 0
                            ):
                                # i -> k -> j
                                cpdag[j, i] = 0  # orient i -> j
                                changed = True
                                break

                    # Rule 4 (Meek's third): i -- j, and two non-adjacent
                    # nodes k, l with i -- k -> j and i -- l -> j: i -> j.
                    if cpdag[i, j] == 1 and cpdag[j, i] == 1:
                        into_j = [
                            k
                            for k in range(d)
                            if k not in (i, j)
                            and cpdag[k, j] == 1
                            and cpdag[j, k] == 0
                            and cpdag[i, k] == 1
                            and cpdag[k, i] == 1
                        ]
                        if any(
                            cpdag[k, m] == 0 and cpdag[m, k] == 0
                            for a, k in enumerate(into_j)
                            for m in into_j[a + 1 :]
                        ):
                            cpdag[j, i] = 0  # orient i -> j
                            changed = True

        return cpdag

    def _ci_test_pval(
        self,
        X: np.ndarray,
        i: int,
        j: int,
        S: list[int],
        n: int,
    ) -> float:
        """
        Conditional independence test: X_i _||_ X_j | X_S.

        Returns p-value.
        """
        if self.ci_test == "fisherz":
            return _fisher_z_test(X, i, j, S, n)
        if self.ci_test in _DISCRETE_TESTS:
            strata = None
            if S:
                strata = np.unique(X[:, S], axis=0, return_inverse=True)[1].reshape(-1)
            stat, dof, _, _ = discrete_ci_test(X[:, i], X[:, j], strata, self.ci_test)
            # Untestable on these data: not rejected.
            return float(sp_stats.chi2.sf(stat, dof)) if dof else 1.0
        raise MethodIncompatibility(
            f"pc_algorithm: ci_test={self.ci_test!r} is not one of "
            f"{('fisherz',) + _DISCRETE_TESTS}."
        )

    def summary(self) -> str:
        """Print a summary of the learned structure."""
        if not hasattr(self, "_cpdag"):
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


def _fisher_z_test(
    X: np.ndarray,
    i: int,
    j: int,
    S: list[int],
    n: int,
) -> float:
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
    r = float(np.clip(r, -1 + 1e-10, 1 - 1e-10))

    # Fisher's Z transformation
    z = 0.5 * np.log((1 + r) / (1 - r))
    # Under H0, sqrt(n - |S| - 3) * z ~ N(0, 1)
    dof = n - len(S) - 3
    if dof < 1:
        return 1.0  # not enough degrees of freedom

    z_stat = np.sqrt(dof) * abs(z)
    pval = 2 * sp_stats.norm.sf(z_stat)
    return float(pval)


def _partial_correlation(
    X: np.ndarray,
    i: int,
    j: int,
    S: list[int],
) -> float:
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
        return float(-P[0, 1] / denom)
    except np.linalg.LinAlgError:
        return 0.0
