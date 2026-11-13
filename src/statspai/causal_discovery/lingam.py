"""LiNGAM — Linear Non-Gaussian Acyclic Model (Shimizu et al., JMLR 2006).

Identifies a causal DAG under the assumptions:

1. Causal structure is linear and acyclic.
2. All disturbances are mutually independent and non-Gaussian.
3. No latent common causes.

Under these assumptions the causal order is **uniquely identifiable**
from purely observational data (in contrast with constraint-based
methods like PC which only identify a Markov-equivalence class).

This implementation uses the DirectLiNGAM algorithm (Shimizu et al. 2011):

    1. Find the variable that is most independent of the residuals of the
       others regressed on it. Independence is measured pair by pair with
       the likelihood-ratio statistic of Hyvarinen and Smith (2013), a
       difference of differential entropies under the maximum-entropy
       approximation of Hyvarinen (1998). Declare that variable a source.
    2. Regress the remaining variables on the source, take residuals.
    3. Recurse on the residuals until all variables are ordered.
    4. The adjacency matrix B[i, j] of direct effects is then recovered
       by regressing each variable on its predecessors in the ordering.

The causal order is the one the ``lingam`` Python package
(``DirectLiNGAM(measure='pwling')``) returns on the same data
(``tests/reference_parity/test_lingam_package_parity.py``). Step 4 is a
plain least-squares fit on all predecessors; the package prunes with an
adaptive Lasso there, so the two coefficient matrices are different
estimates of the same quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin

# --------------------------------------------------------------------- #
#  Helper: pairwise independence measure
# --------------------------------------------------------------------- #


def _entropy_maxent(u: np.ndarray) -> float:
    """Differential entropy of a standardised variable.

    Maximum-entropy approximation of Hyvarinen (1998) with the constants
    of Hyvarinen and Smith (2013, eq. 4): the entropy of a unit-variance
    Gaussian minus two non-negative terms that grow with non-Gaussianity.
    """
    k1 = 79.047
    k2 = 7.4129
    gamma = 0.37457
    return float(
        (1.0 + np.log(2.0 * np.pi)) / 2.0
        - k1 * (np.mean(np.log(np.cosh(u))) - gamma) ** 2
        - k2 * (np.mean(u * np.exp(-(u**2) / 2.0))) ** 2
    )


def _residual(xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
    """Residual of the least-squares regression of ``xi`` on ``xj``."""
    return xi - (np.cov(xi, xj, bias=True)[0, 1] / np.var(xj)) * xj


def _diff_mutual_info(xi_std: np.ndarray, xj_std: np.ndarray) -> float:
    """Likelihood ratio for ``x_i -> x_j`` against ``x_j -> x_i``.

    The difference of the mutual informations ``I(x_j, r_i^(j))`` and
    ``I(x_i, r_j^(i))`` (Hyvarinen and Smith 2013, eq. 3), with ``r_i^(j)``
    the residual of ``x_i`` regressed on ``x_j``. It is positive when
    ``x_i`` is the cause: the residual of the effect on the cause is then
    independent of the cause.
    """
    ri_j = _residual(xi_std, xj_std)
    rj_i = _residual(xj_std, xi_std)
    return (_entropy_maxent(xj_std) + _entropy_maxent(ri_j / np.std(ri_j))) - (
        _entropy_maxent(xi_std) + _entropy_maxent(rj_i / np.std(rj_i))
    )


# --------------------------------------------------------------------- #
#  Result
# --------------------------------------------------------------------- #


@dataclass
class LiNGAMResult(ResultProtocolMixin):
    """Result of a :func:`lingam` (DirectLiNGAM) fit.

    Attributes
    ----------
    order : list of int
        Causal order, most exogenous variable first (column indices).
    adjacency : numpy.ndarray
        ``B[i, j]`` is the direct structural effect of variable ``j`` on ``i``.
    names : list of str
        Variable names, aligned with the columns of ``adjacency``.
    residuals : numpy.ndarray
        ``(n, k)`` structural residuals after removing recovered effects.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> n = 400
    >>> x0 = rng.uniform(-1, 1, n) ** 3
    >>> x1 = 1.5 * x0 + rng.exponential(0.4, n) - 0.4
    >>> df = pd.DataFrame({"x0": x0, "x1": x1})
    >>> res = sp.lingam(df)
    >>> isinstance(res, sp.LiNGAMResult)
    True
    >>> res.to_frame().shape                # adjacency as a labelled DataFrame
    (2, 2)
    >>> bool(len(res.edges(threshold=0.5)) >= 1)
    True
    """

    order: List[int]  # causal order (most exogenous first)
    adjacency: np.ndarray  # B[i, j] = direct effect of j on i
    names: List[str]
    residuals: np.ndarray  # (n, k) final residuals

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.adjacency, index=self.names, columns=self.names)

    def edges(self, threshold: float = 0.05) -> List[tuple]:
        out = []
        for i, ni in enumerate(self.names):
            for j, nj in enumerate(self.names):
                if i == j:
                    continue
                if abs(self.adjacency[i, j]) > threshold:
                    out.append((nj, ni, float(self.adjacency[i, j])))
        return out

    def summary(self) -> str:
        ordered = [self.names[i] for i in self.order]
        lines = [
            "DirectLiNGAM",
            "-" * 40,
            "Causal order (most exogenous first):",
            "  " + " → ".join(ordered),
            "",
            "Adjacency matrix B (B[i, j] = effect of j on i):",
            self.to_frame().round(3).to_string(),
            "",
            "Detected edges (|B| > 0.05):",
        ]
        for src, dst, w in self.edges(threshold=0.05):
            lines.append(f"  {src} → {dst}   β={w:+.3f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# --------------------------------------------------------------------- #
#  Main algorithm
# --------------------------------------------------------------------- #


def _pick_exogenous(X: np.ndarray, remaining: List[int]) -> int:
    """DirectLiNGAM search for the next variable in the causal order.

    For candidate ``i`` the statistic is

        T(i) = sum over j != i of min(0, m(i, j)) ** 2,

    with ``m(i, j)`` the likelihood ratio of :func:`_diff_mutual_info` on
    the standardised pair. ``m(i, j) < 0`` is evidence that ``j`` comes
    before ``i``, so the candidate with the smallest ``T`` is taken
    (Shimizu et al. 2011, Algorithm 1 with the measure of Hyvarinen and
    Smith 2013). Ties go to the first candidate.
    """
    if len(remaining) == 1:
        return remaining[0]
    std = {i: (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i]) for i in remaining}
    scores = []
    for i in remaining:
        total = 0.0
        for j in remaining:
            if j != i:
                total += min(0.0, _diff_mutual_info(std[i], std[j])) ** 2
        scores.append(total)
    return remaining[int(np.argmin(scores))]


def lingam(
    data: Union[pd.DataFrame, np.ndarray],
    standardize: bool = True,
) -> LiNGAMResult:
    """Fit DirectLiNGAM (Shimizu et al. 2011).

    Parameters
    ----------
    data : pd.DataFrame or (n, k) ndarray
    standardize : bool, default True
        Scale each variable to unit variance before the algorithm. The
        variables are centred either way, so ``adjacency`` (OLS slopes in
        the original units) does not depend on the column means. Rows
        with a missing value are dropped.

    Returns
    -------
    LiNGAMResult
        Holds the recovered causal ``order``, the ``adjacency`` matrix of
        direct effects, and the final ``residuals``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> x0 = rng.uniform(-1, 1, n) ** 3            # non-Gaussian source
    >>> x1 = 2.0 * x0 + rng.exponential(0.5, n) - 0.5
    >>> x2 = -1.5 * x1 + rng.uniform(-1, 1, n)
    >>> df = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})
    >>> res = sp.lingam(df)
    >>> [res.names[i] for i in res.order]          # recovered causal order
    ['x0', 'x1', 'x2']
    >>> res.adjacency.shape
    (3, 3)
    """
    if isinstance(data, pd.DataFrame):
        names = list(data.columns)
        labelled = [c for c in names if not pd.api.types.is_numeric_dtype(data[c])]
        if labelled:
            from ..exceptions import MethodIncompatibility

            raise MethodIncompatibility(
                "lingam: the model is linear with non-Gaussian noise and "
                f"needs numeric columns; {labelled} are not.",
                recovery_hint=(
                    "For categorical data use "
                    "sp.pc_algorithm(df, ci_test='chi-square')."
                ),
                diagnostics={"non_numeric_columns": labelled},
                alternative_functions=["sp.pc_algorithm"],
            )
        X = data.to_numpy(dtype=float).copy()
    else:
        X = np.asarray(data, dtype=float).copy()
        names = [f"x{i}" for i in range(X.shape[1])]
    # Listwise deletion, as in the other structure learners; a NaN used to
    # reach np.linalg.lstsq and stop with "SVD did not converge".
    X = X[~np.isnan(X).any(axis=1)]
    # A constant column has no standardised version and an exactly collinear
    # one has a zero residual; neither has a place in a causal order.
    from ._data import check_gaussian_columns

    check_gaussian_columns(X, names, "lingam")
    # Keep original scales so we can de-standardise B at the end.
    scales = X.std(axis=0)
    means = X.mean(axis=0)
    # Centre always: the structural coefficients are slopes, and the
    # regressions below have no intercept. Without centring, a column mean
    # leaked into B whenever standardize=False.
    X = X - means
    if standardize:
        X = X / np.maximum(scales, 1e-12)
    n, k = X.shape
    remaining = list(range(k))
    order: List[int] = []
    current = X.copy()
    while remaining:
        pick = _pick_exogenous(current, remaining)
        order.append(pick)
        # Regress all other remaining on `pick` and replace in current
        src = current[:, pick]
        for j in remaining:
            if j == pick:
                continue
            current[:, j] = _residual(current[:, j], src)
        remaining.remove(pick)

    # Now estimate B by regressing each variable on its predecessors
    # (in the discovered order).
    B = np.zeros((k, k))
    Z = X.copy()
    for rank_idx, i in enumerate(order):
        if rank_idx == 0:
            continue
        predecessors = order[:rank_idx]
        Xp = Z[:, predecessors]
        # OLS of Z[:, i] on Xp
        beta, *_ = np.linalg.lstsq(Xp, Z[:, i], rcond=None)
        for j_local, j_global in enumerate(predecessors):
            B[i, j_global] = float(beta[j_local])

    # Residuals after subtracting recovered structural effects (std scale)
    resid_std = Z - Z @ B.T
    # Map the standardised-scale B back to original scale:
    #   B_orig[i, j] = B_std[i, j] * sigma_i / sigma_j
    if standardize:
        B_orig = np.zeros_like(B)
        for i in range(k):
            for j in range(k):
                B_orig[i, j] = B[i, j] * scales[i] / max(scales[j], 1e-12)
        B = B_orig
        resid = resid_std * scales  # rescale to original units
    else:
        resid = resid_std

    _result = LiNGAMResult(
        order=order,
        adjacency=B,
        names=names,
        residuals=resid,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.causal_discovery.lingam",
            params={"standardize": standardize},
            data=data if hasattr(data, "columns") else None,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
