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
Advances in Neural Information Processing Systems, 31.
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import expm


# ======================================================================
# Public API
# ======================================================================

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
        'n_edges' : int
        'n_obs' : int
        'lambda1' : float
        'w_threshold' : float

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.notears(df, variables=['X', 'Z', 'M', 'Y'])
    >>> print(result['edges'])  # [(parent, child, weight), ...]
    >>> print(result['adjacency'])  # weighted adjacency matrix
    """
    est = NOTEARS(
        data=data, variables=variables, lambda1=lambda1,
        max_iter=max_iter, h_tol=h_tol, rho_max=rho_max,
        w_threshold=w_threshold, random_state=random_state,
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
    ):
        self.data = data
        self.variables = variables
        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.random_state = random_state

    def fit(self) -> Dict[str, Any]:
        """Run NOTEARS and return the learned DAG."""
        np.random.seed(self.random_state)

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
            raise ValueError("At least 2 variables are required for DAG learning")

        # Standardise
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        # Run NOTEARS
        W_est = self._notears_linear(X, d, n)

        # Threshold small weights
        W_est[np.abs(W_est) < self.w_threshold] = 0.0

        # Compute final acyclicity
        h_val = float(_h_func(W_est))

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

        return {
            'adjacency': adj_df,
            'edges': edges,
            'dag': dag_df,
            'variables': var_names,
            'h_value': h_val,
            'n_edges': len(edges),
            'n_obs': n,
            'lambda1': self.lambda1,
            'w_threshold': self.w_threshold,
        }

    def _notears_linear(self, X, d, n):
        """
        Solve the NOTEARS optimisation via augmented Lagrangian.

        min_{W}  F(W) + lambda1 * ||W||_1
        s.t.     h(W) = 0

        Augmented Lagrangian: L(W, alpha, rho) = F(W) + lambda1*||W||_1
                              + alpha * h(W) + 0.5 * rho * h(W)^2
        """
        # Initialise
        W = np.zeros((d, d))
        rho = 1.0
        alpha = 0.0
        h_prev = np.inf

        for iteration in range(self.max_iter):
            # Inner optimisation: minimise augmented Lagrangian over W
            W = self._solve_inner(X, W, rho, alpha, d, n)

            # Evaluate acyclicity constraint
            h_new = _h_func(W)

            if h_new > 0.25 * h_prev:
                rho *= 10.0
            else:
                pass  # rho stays the same

            alpha += rho * h_new
            h_prev = h_new

            if h_new <= self.h_tol:
                break

            if rho > self.rho_max:
                break

        return W

    def _solve_inner(self, X, W_init, rho, alpha, d, n):
        """Solve inner L-BFGS-B problem for fixed rho, alpha."""

        def _objective(w_flat):
            W = w_flat.reshape(d, d)

            # Least-squares loss: 0.5/n * ||X - X@W||_F^2
            R = X - X @ W
            loss = 0.5 / n * np.sum(R ** 2)

            # Acyclicity
            h = _h_func(W)

            # Augmented Lagrangian terms
            obj = loss + alpha * h + 0.5 * rho * h * h

            return obj

        def _gradient(w_flat):
            W = w_flat.reshape(d, d)

            # Gradient of least-squares loss
            R = X - X @ W
            grad_loss = -1.0 / n * X.T @ R

            # Gradient of h(W) = tr(e^{W o W}) - d
            # dh/dW = (e^{W o W})^T o 2W
            M = W * W  # element-wise square
            E = expm(M)
            grad_h = E.T * 2 * W

            grad = grad_loss + (alpha + rho * _h_func(W)) * grad_h

            return grad.ravel()

        # Use L-BFGS-B with bounds to implement L1 via variable splitting
        # For simplicity, use proximal gradient for L1
        # Here we use a smooth approximation: add lambda1 * sum |W_ij|
        # via subgradient in L-BFGS-B bounds

        # Actually, the standard NOTEARS approach: solve without L1 in
        # L-BFGS-B, then threshold. For L1, we can use the variable
        # splitting trick: W = W+ - W-, W+,W- >= 0
        d2 = d * d

        if self.lambda1 > 0:
            # Variable splitting: w = [w+; w-], W = W+ - W-
            w0 = np.zeros(2 * d2)
            w_init = W_init.ravel()
            w0[:d2] = np.maximum(w_init, 0)
            w0[d2:] = np.maximum(-w_init, 0)

            bounds = [(0, None)] * (2 * d2)

            def _obj_split(w):
                wp = w[:d2].reshape(d, d)
                wm = w[d2:].reshape(d, d)
                W = wp - wm
                R = X - X @ W
                loss = 0.5 / n * np.sum(R ** 2)
                h = _h_func(W)
                l1 = self.lambda1 * (np.sum(wp) + np.sum(wm))
                return loss + alpha * h + 0.5 * rho * h * h + l1

            def _grad_split(w):
                wp = w[:d2].reshape(d, d)
                wm = w[d2:].reshape(d, d)
                W = wp - wm
                R = X - X @ W
                grad_loss = -1.0 / n * X.T @ R
                M = W * W
                E = expm(M)
                h = float(np.trace(E) - d)
                grad_h = E.T * 2 * W
                grad_W = grad_loss + (alpha + rho * h) * grad_h

                grad = np.zeros(2 * d2)
                grad[:d2] = (grad_W + self.lambda1).ravel()
                grad[d2:] = (-grad_W + self.lambda1).ravel()
                return grad

            result = minimize(
                _obj_split, w0,
                jac=_grad_split,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12},
            )
            w_opt = result.x
            W_opt = w_opt[:d2].reshape(d, d) - w_opt[d2:].reshape(d, d)
        else:
            w0 = W_init.ravel()
            result = minimize(
                _objective, w0,
                jac=_gradient,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12},
            )
            W_opt = result.x.reshape(d, d)

        # Zero out diagonal (no self-loops)
        np.fill_diagonal(W_opt, 0)

        return W_opt

    def summary(self) -> str:
        """Print a summary of the learned DAG."""
        if not hasattr(self, '_W'):
            raise ValueError("Model must be fitted first. Call .fit()")

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
                    edges.append((self._var_names[i], self._var_names[j],
                                  self._W[i, j]))
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

def _h_func(W):
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
