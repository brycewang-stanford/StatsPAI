"""
NOTEARS: DAGs with NO TEARS — Continuous Optimization for Structure Learning.

Learns a directed acyclic graph (DAG) from observational data by solving:

    min_{W}  0.5/n * ||X - X @ W||_F^2  +  lambda1 * ||W||_1
    s.t.     h(W) = tr(e^{W o W}) - d = 0   (acyclicity)

where W is the weighted adjacency matrix, h(W) is the acyclicity
constraint (Zheng et al. 2018), and lambda1 is the L1 sparsity penalty.

Solved via augmented Lagrangian method with L-BFGS-B inner optimiser.

References
----------
Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018).
"DAGs with NO TEARS: Continuous Optimization for Structure Learning."
Advances in Neural Information Processing Systems, 31. [@zheng2018dags]
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

from .._aliases import accepts_aliases
from .._input_validation import check_columns
from ..utils._rng import preserve_global_rng

# ======================================================================
# Public API
# ======================================================================


@accepts_aliases(seed="random_state")
def notears(
    data: pd.DataFrame,
    variables: Optional[List[str]] = None,
    lambda1: float = 0.1,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    w_threshold: float = 0.3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Learn a DAG from data using NOTEARS (Zheng et al. 2018).

    Parameters
    ----------
    data : pd.DataFrame
        Observational data (n_samples x d_variables).
    variables : list of str, optional
        Column names to use. If None, uses all numeric columns.
    lambda1 : float, default 0.1
        L1 penalty weight for sparsity. Higher = sparser graph.
    max_iter : int, default 100
        Maximum augmented Lagrangian iterations.
    h_tol : float, default 1e-8
        Convergence threshold for acyclicity constraint h(W).
    rho_max : float, default 1e16
        Maximum penalty parameter rho.
    w_threshold : float, default 0.3
        Threshold for pruning small edge weights. Edges with
        |W_ij| < w_threshold are removed.
    random_state : int, default 42
        Random seed.

    Returns
    -------
    dict
        'adjacency' : pd.DataFrame
            Weighted adjacency matrix W (W_ij means i -> j).
        'edges' : list of (str, str, float)
            List of (parent, child, weight) tuples.
        'dag' : pd.DataFrame
            Binary adjacency matrix (thresholded).
        'variables' : list of str
        'h_value' : float
            Final acyclicity constraint value (should be ~0).
        'is_dag' : bool
            Whether the thresholded graph is acyclic. When the optimiser
            stops early it may not be, and a ``RuntimeWarning`` says so.
        'n_edges' : int
        'n_obs' : int
        'lambda1' : float
        'w_threshold' : float

    Notes
    -----
    The columns are standardised before fitting, and the weights are on
    that scale. Read the output as a skeleton with weights, not as
    evidence of direction: with linear Gaussian data the least-squares
    score cannot tell Markov-equivalent graphs apart except through the
    variables' scales, and standardising removes those. On the chain
    ``a -> b -> c`` with unit noise the two adjacencies are found in every
    run, and the arrows come out as ``b -> a``, ``c -> b`` or ``b -> a``,
    ``b -> c`` depending on the sample.

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
    >>> result = sp.notears(df, variables=['X', 'Z', 'M', 'Y'])
    >>> bool(result['n_edges'] >= 0)  # learned edge count
    True
    """
    est = NOTEARS(
        data=data,
        variables=variables,
        lambda1=lambda1,
        max_iter=max_iter,
        h_tol=h_tol,
        rho_max=rho_max,
        w_threshold=w_threshold,
        random_state=random_state,
    )
    return est.fit()


# ======================================================================
# NOTEARS Estimator
# ======================================================================


class NOTEARS:
    """
    NOTEARS: Continuous optimization for DAG structure learning.

    Parameters
    ----------
    data : pd.DataFrame
    variables : list of str, optional
    lambda1 : float
    max_iter : int
    h_tol : float
    rho_max : float
    w_threshold : float
    random_state : int

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> from statspai.causal_discovery.notears import NOTEARS
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> X = rng.normal(size=n)
    >>> Z = 0.8 * X + rng.normal(size=n) * 0.5
    >>> M = 0.7 * Z + rng.normal(size=n) * 0.5
    >>> Y = 0.6 * M + rng.normal(size=n) * 0.5
    >>> df = pd.DataFrame({'X': X, 'Z': Z, 'M': M, 'Y': Y})
    >>> est = NOTEARS(data=df, variables=['X', 'Z', 'M', 'Y'])
    >>> result = est.fit()
    >>> bool(result['n_edges'] >= 0)
    True

    References
    ----------
    [@zheng2018dags]
    """

    def __init__(
        self,
        data: pd.DataFrame,
        variables: Optional[List[str]] = None,
        lambda1: float = 0.1,
        max_iter: int = 100,
        h_tol: float = 1e-8,
        rho_max: float = 1e16,
        w_threshold: float = 0.3,
        random_state: int = 42,
    ) -> None:
        self.data = data
        self.variables = variables
        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.random_state = random_state

    @preserve_global_rng
    def fit(self) -> Dict[str, Any]:
        """Run NOTEARS and return the learned DAG."""
        np.random.seed(self.random_state)

        # Prepare data
        if self.variables is not None:
            check_columns(
                self.data,
                {"variables": list(self.variables)},
                function="sp.notears",
            )
            X = self.data[self.variables].dropna().values.astype(np.float64)
            var_names = list(self.variables)
        else:
            from ._data import numeric_columns

            numeric = self.data[numeric_columns(self.data)]
            X = numeric.dropna().values.astype(np.float64)
            var_names = list(numeric.columns)

        n, d = X.shape
        if d < 2:
            dropped = [c for c in self.data.columns if c not in var_names]
            raise ValueError(
                "At least 2 variables are required for DAG learning"
                + (
                    f"; {len(dropped)} non-numeric column(s) were left out "
                    f"({dropped[:6]}). NOTEARS is a linear model of numeric "
                    "columns: for categorical data use "
                    "sp.pc_algorithm(df, ci_test='chi-square')."
                    if dropped
                    else "."
                )
            )

        # Standardise
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        # Run NOTEARS
        W_est = self._notears_linear(X, d, n)

        # Threshold small weights
        W_est[np.abs(W_est) < self.w_threshold] = 0.0

        # Compute final acyclicity
        h_val = float(_h_func(W_est))
        from ._data import warn_if_cyclic

        is_dag = warn_if_cyclic(W_est, "notears", h_val)

        # Build results
        adj_df = pd.DataFrame(W_est, index=var_names, columns=var_names)

        edges = []
        for i in range(d):
            for j in range(d):
                if W_est[i, j] != 0:
                    edges.append((var_names[i], var_names[j], float(W_est[i, j])))
        edges.sort(key=lambda e: abs(e[2]), reverse=True)

        dag_binary = (np.abs(W_est) > 0).astype(int)
        dag_df = pd.DataFrame(dag_binary, index=var_names, columns=var_names)

        self._W = W_est
        self._var_names = var_names

        from ._viz import DAGDict

        return DAGDict(
            {
                "adjacency": adj_df,
                "edges": edges,
                "dag": dag_df,
                "variables": var_names,
                "h_value": h_val,
                "is_dag": is_dag,
                "n_edges": len(edges),
                "n_obs": n,
                "lambda1": self.lambda1,
                "w_threshold": self.w_threshold,
            }
        )

    def _notears_linear(
        self,
        X: np.ndarray,
        d: int,
        n: int,
    ) -> np.ndarray:
        """
        Solve the NOTEARS optimisation via augmented Lagrangian.

        min_{W}  F(W) + lambda1 * ||W||_1
        s.t.     h(W) = 0

        Augmented Lagrangian: L(W, alpha, rho) = F(W) + lambda1*||W||_1
                              + alpha * h(W) + 0.5 * rho * h(W)^2

        Algorithm 1 of Zheng et al. (2018). Each outer step solves the
        primal problem, raising ``rho`` tenfold and solving again until the
        constraint has fallen to a quarter of its last accepted value, and
        only then takes the dual step ``alpha += rho * h``. The L1 term is
        made smooth by splitting ``W = W+ - W-`` with both parts
        non-negative, and the split variables are carried from one solve
        to the next.

        Through 1.43.1 every primal solution was accepted and the dual
        step was taken with the raised ``rho`` even when the constraint
        had not fallen, which is not the published scheme and ended at
        other graphs than the reference implementation on a third of the
        test designs.
        """
        w = np.zeros(2 * d * d)
        rho = 1.0
        alpha = 0.0
        h = np.inf
        for _ in range(self.max_iter):
            w_new = w
            h_new = h
            while rho < self.rho_max:
                w_new = self._solve_inner(X, w, rho, alpha, d, n)
                h_new = _h_func(_unsplit(w_new, d))
                if h_new > 0.25 * h:
                    rho *= 10.0
                else:
                    break
            w, h = w_new, h_new
            alpha += rho * h
            if h <= self.h_tol or rho >= self.rho_max:
                break
        return _unsplit(w, d)

    def _solve_inner(
        self,
        X: np.ndarray,
        w_init: np.ndarray,
        rho: float,
        alpha: float,
        d: int,
        n: int,
    ) -> np.ndarray:
        """Minimise the augmented Lagrangian over the split variables
        ``[W+, W-]`` for fixed ``rho`` and ``alpha`` (L-BFGS-B)."""
        d2 = d * d
        lam = self.lambda1
        # No self-loops: the diagonal of both parts is held at zero.
        off_diag = ~np.eye(d, dtype=bool).ravel()
        bounds = [(0.0, None) if off else (0.0, 0.0) for off in off_diag] * 2

        def _func(w: np.ndarray) -> tuple[float, np.ndarray]:
            W = _unsplit(w, d)
            R = X - X @ W
            loss = 0.5 / n * float(np.sum(R**2))
            grad_loss = -1.0 / n * X.T @ R
            # h(W) = tr(e^{W o W}) - d and its gradient (e^{W o W})^T o 2W
            E = expm(W * W)
            h = float(np.trace(E) - d)
            grad_W = grad_loss + (alpha + rho * h) * (E.T * 2 * W)
            obj = loss + alpha * h + 0.5 * rho * h * h + lam * float(w.sum())
            grad = np.concatenate([(grad_W + lam).ravel(), (-grad_W + lam).ravel()])
            return obj, grad

        result = minimize(_func, w_init, jac=True, method="L-BFGS-B", bounds=bounds)
        return np.asarray(result.x, dtype=float)

    def summary(self) -> str:
        """Print a summary of the learned DAG."""
        if not hasattr(self, "_W"):
            raise ValueError(
                "NOTEARS.summary(): Model must be fitted first. Call .fit()"
            )

        lines = []
        lines.append("=" * 60)
        lines.append("  NOTEARS: DAG Structure Learning")
        lines.append("  Zheng et al. (2018)")
        lines.append("=" * 60)
        lines.append(f"  Variables: {', '.join(self._var_names)}")
        lines.append(f"  Lambda1 (sparsity): {self.lambda1}")
        lines.append(f"  Threshold: {self.w_threshold}")

        edges = []
        d = len(self._var_names)
        for i in range(d):
            for j in range(d):
                if self._W[i, j] != 0:
                    edges.append(
                        (self._var_names[i], self._var_names[j], self._W[i, j])
                    )
        edges.sort(key=lambda e: abs(e[2]), reverse=True)

        lines.append(f"  Edges found: {len(edges)}")
        lines.append("")
        lines.append("  Learned Edges:")
        lines.append("  " + "-" * 40)
        for parent, child, w in edges:
            lines.append(f"    {parent} -> {child}  (weight: {w:.4f})")
        lines.append("=" * 60)

        return "\n".join(lines)


# ======================================================================
# Acyclicity constraint
# ======================================================================


def _unsplit(w: np.ndarray, d: int) -> np.ndarray:
    """``W = W+ - W-`` from the stacked non-negative parts."""
    return np.asarray((w[: d * d] - w[d * d :]).reshape(d, d), dtype=float)


def _h_func(W: np.ndarray) -> float:
    """
    Acyclicity constraint: h(W) = tr(e^{W o W}) - d.

    h(W) = 0 iff W encodes a DAG.

    Parameters
    ----------
    W : np.ndarray
        d x d weighted adjacency matrix.

    Returns
    -------
    float
    """
    d = W.shape[0]
    M = W * W  # element-wise (Hadamard) square
    E = expm(M)
    return float(np.trace(E) - d)
