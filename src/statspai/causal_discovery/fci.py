"""
FCI (Fast Causal Inference) algorithm for causal discovery with
possibly *unobserved* confounders (Spirtes, Meek & Richardson 1995;
Zhang 2008).

The output is a **Partial Ancestral Graph (PAG)** — a mixed graph whose
edge marks are:

=====  =================================================================
mark   meaning
=====  =================================================================
``o``  uncertain (circle) — could be a tail or an arrowhead
``>``  arrowhead — the incident node is *not* an ancestor of the other
``-``  tail — the incident node *is* an ancestor of the other
=====  =================================================================

So for nodes :math:`X, Y`:

* ``X --> Y`` : X is a cause of Y (no latent in between)
* ``X <-> Y`` : latent common cause (bidirected)
* ``X o-> Y`` : Y is not an ancestor of X, but X could be a cause of Y
  or share a latent
* ``X o-o Y`` : no orientation determined

The algorithm, as in ``pcalg::fci``:

1. **Skeleton**: the PC-stable search of ``sp.pc_algorithm``.
2. **Possible-D-SEP**: with latent variables a separating set need not lie
   among the neighbours of either node. Colliders are oriented on the
   skeleton and each remaining edge is tested given subsets of the nodes
   reachable along collider-or-triangle paths. ``possible_dsep=False``
   skips this pass (the skeleton of RFCI, [@colombo2012learning]).
3. **Orientation**: every edge back to ``o-o``, colliders on unshielded
   triples, then Zhang's rules R1-R10 to a fixed point.

With ``ci_test='fisherz'`` the PAG equals that of ``pcalg::fci(indepTest =
gaussCItest)`` mark for mark on the 24 reference data sets of
``tests/reference_parity/test_fci_pcalg_parity.py``.

References
----------
Spirtes, P., Meek, C., & Richardson, T. (1995).
"Causal inference in the presence of latent variables and selection
bias." *UAI-95*, 499-506.

Zhang, J. (2008).
"On the completeness of orientation rules for causal discovery in the
presence of latent confounders and selection bias." *Artificial
Intelligence*, 172(16-17), 1873-1896. [@zhang2008completeness]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin

# Edge-mark constants used in the mark matrix.
MARK_NONE = 0  # no edge
MARK_CIRCLE = 1  # 'o'
MARK_ARROW = 2  # '>'
MARK_TAIL = 3  # '-'

_MARK_SYMBOL = {
    MARK_NONE: ".",
    MARK_CIRCLE: "o",
    MARK_ARROW: ">",
    MARK_TAIL: "-",
}

#: Marks as drawn at the left end of an edge.
_LEFT_SYMBOL = {
    MARK_NONE: ".",
    MARK_CIRCLE: "o",
    MARK_ARROW: "<",
    MARK_TAIL: "-",
}


@dataclass
class FCIResult(ResultProtocolMixin):
    """Partial Ancestral Graph (PAG) learned by :func:`fci`.

    Attributes
    ----------
    variables : list of str
        Variable names (graph nodes).
    skeleton : pd.DataFrame
        Undirected adjacency matrix over ``variables``.
    pag_left, pag_right : pd.DataFrame
        Edge marks on the i-side and j-side of each edge (i, j).
    edges : list of tuple
        Human-readable ``(i, label, j)`` edges, e.g. ``("X", "-->", "Y")``.
    separating_sets : dict
        CI-test separating sets keyed by variable-name pairs.
    n_obs : int
        Number of complete observations used.
    alpha : float
        Significance level of the CI tests.
    ci_test : str
        Name of the conditional-independence test.
    n_removed_by_possible_dsep : int
        Edges of the PC skeleton that the Possible-D-SEP pass removed.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> x = rng.normal(size=n)
    >>> m = x + rng.normal(size=n)
    >>> y = m + rng.normal(size=n)
    >>> data = pd.DataFrame({"X": x, "M": m, "Y": y})
    >>> res = sp.fci(data)
    >>> bool(res.skeleton.shape == (3, 3))
    True
    """

    variables: List[str]
    skeleton: pd.DataFrame
    pag_left: pd.DataFrame  # marks on the i-side of edge (i,j)
    pag_right: pd.DataFrame  # marks on the j-side of edge (i,j)
    edges: List[Tuple[str, str, str]]  # (i, label, j) e.g. ("X", "-->", "Y")
    separating_sets: Dict[Tuple[str, str], Set[str]]
    n_obs: int
    alpha: float
    ci_test: str
    n_removed_by_possible_dsep: int = 0

    def summary(self) -> str:  # pragma: no cover
        lines = ["FCI / PAG edges:"]
        for i, lab, j in self.edges:
            lines.append(f"  {i} {lab} {j}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return f"FCIResult(d={len(self.variables)}, edges={len(self.edges)})"


# --------------------------------------------------------------------
# CI test
# --------------------------------------------------------------------


def _fisher_z(X: np.ndarray, i: int, j: int, S: Sequence[int], n: int) -> float:
    """Partial-correlation Fisher-Z test; returns p-value of H0: indep."""
    idx = [i, j] + list(S)
    sub = X[:, idx]
    corr = np.corrcoef(sub, rowvar=False)
    try:
        precision = np.linalg.pinv(corr)
    except np.linalg.LinAlgError:
        return 1.0
    denom = np.sqrt(precision[0, 0] * precision[1, 1])
    if denom <= 0:
        return 1.0
    pcorr = -precision[0, 1] / denom
    pcorr = np.clip(pcorr, -0.999999, 0.999999)
    z = 0.5 * np.log((1 + pcorr) / (1 - pcorr))
    stat = np.sqrt(max(n - len(S) - 3, 1)) * abs(z)
    return float(2.0 * stats.norm.sf(stat))


# --------------------------------------------------------------------
# Skeleton learning
# --------------------------------------------------------------------


def _learn_skeleton(
    X: np.ndarray, alpha: float, max_cond_size: Optional[int]
) -> Tuple[np.ndarray, Dict[Tuple[int, int], Set[int]]]:
    n, d = X.shape
    adj = np.ones((d, d), dtype=int)
    np.fill_diagonal(adj, 0)
    sep_sets: Dict[Tuple[int, int], Set[int]] = {}
    max_k = max_cond_size if max_cond_size is not None else d - 2

    # The same search as sp.pc_algorithm. The loop that stood here stopped
    # at the first level that removed no edge, so an edge whose separating
    # set was two sizes larger than anything found so far was never tested
    # against it: X <- {A, B, C} -> Y kept a spurious X - Y edge at any
    # sample size.
    from .pc import stable_skeleton

    stable_skeleton(
        adj, sep_sets, max_k, alpha, lambda x, y, S: _fisher_z(X, x, y, S, n)
    )
    return adj, sep_sets


def fci(
    data: pd.DataFrame,
    variables: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    max_cond_size: Optional[int] = None,
    ci_test: str = "fisherz",
    possible_dsep: bool = True,
) -> FCIResult:
    """
    Run FCI. Returns a :class:`FCIResult` with the learned PAG.

    Parameters
    ----------
    data : pd.DataFrame
    variables : sequence of str, optional
        Columns to use; defaults to all numeric columns.
    alpha : float, default 0.05
        Significance level for CI tests.
    max_cond_size : int, optional
        Max size of conditioning set.
    ci_test : {"fisherz"}
        Only Fisher-Z partial-correlation test is supported; extensions
        (kernel / chi-square) can be added later.
    possible_dsep : bool, default True
        Run the Possible-D-SEP pass, which FCI needs to be correct when
        there are latent common causes: without it an edge can survive
        that no subset of either node's neighbours separates but a larger
        set does. ``False`` keeps the PC skeleton, which is faster on wide
        data and is what this function returned before 1.39.

    Returns
    -------
    FCIResult
        Learned PAG with skeleton, edge marks, and human-readable edge
        list.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> x = rng.normal(size=n)
    >>> m = x + rng.normal(size=n)
    >>> y = m + rng.normal(size=n)
    >>> data = pd.DataFrame({"X": x, "M": m, "Y": y})
    >>> res = sp.fci(data, alpha=0.05)
    >>> res.variables
    ['X', 'M', 'Y']
    >>> bool(len(res.edges) >= 1)
    True

    References
    ----------
    zhang2008completeness
    """
    if ci_test != "fisherz":
        raise NotImplementedError("Only 'fisherz' is supported at the moment")

    if variables is None:
        variables = list(data.select_dtypes(include=[np.number]).columns)
    variables = list(variables)
    X = data[variables].dropna().to_numpy(dtype=float)
    n = X.shape[0]
    d = X.shape[1]
    if d < 2:
        dropped = [c for c in data.columns if c not in variables]
        raise ValueError(
            "Need at least 2 variables"
            + (
                f"; {len(dropped)} non-numeric column(s) were left out "
                f"({dropped[:6]}). FCI here tests partial correlations: for "
                "categorical data without latent variables use "
                "sp.pc_algorithm(df, ci_test='chi-square')."
                if dropped
                else "."
            )
        )

    from ._fci_core import orient_pag, possible_dsep_removal

    adj, sep_sets = _learn_skeleton(X, alpha, max_cond_size)
    n_removed = 0
    if possible_dsep:
        n_removed = possible_dsep_removal(
            adj,
            sep_sets,
            alpha,
            lambda x, y, S: _fisher_z(X, x, y, S, n),
            max_cond_size,
        )
    # marks[i, j] is the mark at the j end of the edge between i and j
    marks = orient_pag(adj, sep_sets)
    left, right = marks.T.copy(), marks.copy()

    # Build human-readable edge list
    edges: List[Tuple[str, str, str]] = []
    seen = set()
    for i in range(d):
        for j in range(i + 1, d):
            if marks[i, j] == MARK_NONE:
                continue
            li, lj = int(left[i, j]), int(right[i, j])
            # An arrowhead at the left end is drawn '<'. It used to be drawn
            # '>' like the right one, so a bidirected edge read 'X >-> Y'
            # and 'X <-- Y' read 'X >-- Y'.
            arrow = f"{_LEFT_SYMBOL[li]}-{_MARK_SYMBOL[lj]}"
            edges.append((variables[i], arrow, variables[j]))
            seen.add((i, j))

    skeleton_df = pd.DataFrame(adj, index=variables, columns=variables)
    left_df = pd.DataFrame(left, index=variables, columns=variables)
    right_df = pd.DataFrame(right, index=variables, columns=variables)

    sep_named = {
        (variables[i], variables[j]): {variables[k] for k in s}
        for (i, j), s in sep_sets.items()
    }

    _result = FCIResult(
        variables=variables,
        skeleton=skeleton_df,
        pag_left=left_df,
        pag_right=right_df,
        edges=edges,
        separating_sets=sep_named,
        n_obs=n,
        alpha=alpha,
        ci_test=ci_test,
        n_removed_by_possible_dsep=int(n_removed),
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.causal_discovery.fci",
            params={
                "variables": list(variables) if variables else None,
                "alpha": alpha,
                "max_cond_size": max_cond_size,
                "ci_test": ci_test,
                "possible_dsep": possible_dsep,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


__all__ = ["fci", "FCIResult"]
