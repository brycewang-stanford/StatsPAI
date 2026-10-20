"""IDA: causal effects that are possible under a CPDAG.

When causal discovery returns a Markov equivalence class (a CPDAG), the
effect of ``x`` on ``y`` is identified within each DAG of the class but
may differ between them. IDA (intervention calculus when the DAG is
absent) lists the effect under every admissible parent set of ``x``, for
linear-Gaussian models the coefficient of ``x`` in the regression of
``y`` on ``x`` and that parent set (zero when ``y`` is in it).

``method='global'`` enumerates every DAG of the class (one effect per
DAG, so multiplicities follow the class). ``method='local'`` takes ``x``'s
directed parents plus each subset ``S`` of its undirected neighbours whose
orientation into ``x`` creates no new v-structure at ``x`` (every member
of ``S`` is adjacent to every other parent), one effect per admissible
set; the two give the same distinct values. Both equal R ``pcalg::ida``
(GPL, used as a black box) to 1e-10
(``tests/reference_parity/test_ida_pcalg_parity.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["ida", "IDAResult"]

#: Most undirected edges ``method='global'`` enumerates orientations of.
MAX_UNDIRECTED_GLOBAL = 18


@dataclass
class IDAResult(ResultProtocolMixin):
    """Possible effects of ``x`` on ``y`` across a Markov equivalence class.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> cov = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["a", "b"],
    ...                    columns=["a", "b"])
    >>> cp = pd.DataFrame([[0, 1], [1, 0]], index=["a", "b"], columns=["a", "b"])
    >>> r = sp.ida(cov, cp, x="a", y="b")
    >>> [round(e, 2) for e in r.effects]   # a -> b, or b -> a (no effect)
    [0.5, 0.0]
    """

    x: str
    y: str
    effects: List[float]
    parent_sets: List[List[str]]
    method: str
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def min_abs_effect(self) -> float:
        """The smallest absolute effect: a lower bound on the causal effect's
        size that holds whichever DAG of the class is true."""
        return float(np.min(np.abs(self.effects)))

    def to_dict(self) -> dict:
        from .._result_serialize import result_to_dict

        return result_to_dict(self)

    def summary(self) -> str:  # pragma: no cover - formatting
        vals = ", ".join(f"{e:+.4f}" for e in self.effects)
        return (
            f"IDA ({self.method}): possible effects of {self.x} on {self.y}\n"
            f"  {vals}\n  min |effect| = {self.min_abs_effect:.4f}"
        )


def _cpdag_matrix(cpdag: Any, names: Optional[Sequence[str]]) -> pd.DataFrame:
    if isinstance(cpdag, dict) and "cpdag" in cpdag:
        cpdag = cpdag["cpdag"]
    if (
        isinstance(cpdag, dict)
        and cpdag
        and all(isinstance(v, dict) for v in cpdag.values())
    ):
        # DataFrame.to_dict() form, {column: {row: value}}: what an agent
        # sends over JSON (the MCP path refused it, 2026-10-20 probe)
        cpdag = pd.DataFrame(cpdag)
        cpdag = cpdag.reindex(index=list(cpdag.columns))
    if isinstance(cpdag, pd.DataFrame):
        A = cpdag.copy()
    else:
        arr = np.asarray(cpdag)
        if names is None:
            raise MethodIncompatibility(
                "ida: a CPDAG given as an array needs node names.",
                recovery_hint="Pass a DataFrame (sp.pc_algorithm(...)['cpdag']).",
            )
        A = pd.DataFrame(arr, index=list(names), columns=list(names))
    A = (A != 0).astype(int)
    if list(A.index) != list(A.columns):
        raise MethodIncompatibility(
            "ida: the CPDAG's rows and columns must name the same nodes.",
            recovery_hint="Use the 'cpdag' DataFrame from sp.pc_algorithm.",
        )
    return A


def _vstructures(M: np.ndarray) -> set:
    p = M.shape[0]
    out = set()
    for c in range(p):
        pa = [i for i in range(p) if M[i, c] and not M[c, i]]
        for a, b in combinations(pa, 2):
            if not M[a, b] and not M[b, a]:
                out.add((a, b, c))
    return out


def _acyclic(M: np.ndarray) -> bool:
    indeg = M.sum(axis=0).astype(int)
    stack = [i for i in range(len(M)) if indeg[i] == 0]
    seen = 0
    while stack:
        i = stack.pop()
        seen += 1
        for j in np.flatnonzero(M[i]):
            indeg[j] -= 1
            if indeg[j] == 0:
                stack.append(j)
    return seen == len(M)


def _ols_coef(S: np.ndarray, x: int, y: int, Z: Sequence[int]) -> float:
    if y in Z:
        return 0.0
    idx = [x, *sorted(Z)]
    sub = S[np.ix_(idx, idx)]
    return float(np.linalg.solve(sub, S[idx, y])[0])


def ida(
    data: Union[pd.DataFrame, np.ndarray],
    cpdag: Any,
    x: str,
    y: str,
    method: str = "local",
    names: Optional[Sequence[str]] = None,
) -> IDAResult:
    """Possible causal effects of ``x`` on ``y`` under a CPDAG (IDA).

    Parameters
    ----------
    data : pd.DataFrame or ndarray
        The observations (their sample covariance, divisor ``n - 1``, is
        used), or a covariance matrix when ``data`` is square and
        ``names`` / the CPDAG's labels say so: pass a DataFrame whose
        index equals its columns to give a covariance matrix directly.
    cpdag : pd.DataFrame, dict or ndarray
        CPDAG adjacency with ``A[i, j] = 1`` for ``i -> j`` and both
        entries for an undirected edge -- the ``'cpdag'`` of
        ``sp.pc_algorithm`` (the dict itself is accepted) or ``sp.ges``.
    x, y : str
        Cause and effect variables.
    method : {'local', 'global'}, default 'local'
        ``'local'``: one effect per admissible parent set of ``x``.
        ``'global'``: one effect per DAG in the equivalence class.
    names : sequence of str, optional
        Node names when ``cpdag`` is an array.

    Returns
    -------
    IDAResult
        ``effects`` (one per parent set or DAG), ``parent_sets`` and
        ``min_abs_effect``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> a = rng.normal(size=2000); b = a + rng.normal(size=2000)
    >>> c = b + rng.normal(size=2000)
    >>> df = pd.DataFrame({"a": a, "b": b, "c": c})
    >>> cp = pd.DataFrame([[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    ...                   index=list("abc"), columns=list("abc"))
    >>> r = sp.ida(df, cp, x="b", y="c")
    >>> len(r.effects)
    3
    """
    if method not in ("local", "global"):
        raise MethodIncompatibility(
            f"ida: method must be 'local' or 'global', got {method!r}.",
            recovery_hint="Use method='local' (fast) or 'global'.",
        )
    A = _cpdag_matrix(cpdag, names)
    nodes = list(A.index)
    for v in (x, y):
        if v not in nodes:
            raise MethodIncompatibility(
                f"ida: {v!r} is not a node of the CPDAG.",
                recovery_hint=f"Nodes: {nodes}",
            )
    if x == y:
        raise MethodIncompatibility(
            "ida: x and y are the same variable.", recovery_hint="Name two nodes."
        )
    if (
        isinstance(data, pd.DataFrame)
        and list(data.index) == list(data.columns)
        and set(nodes) <= set(data.columns)
        and len(data) == data.shape[1]
    ):
        S = data.loc[nodes, nodes].to_numpy(dtype=float)
        n_obs = None
    else:
        frame = (
            data[nodes]
            if isinstance(data, pd.DataFrame)
            else pd.DataFrame(np.asarray(data, dtype=float), columns=nodes)
        )
        frame = frame.dropna()
        if len(frame) <= len(nodes):
            raise DataInsufficient(
                f"ida: {len(frame)} complete rows for {len(nodes)} variables.",
                recovery_hint="Use more observations or fewer variables.",
            )
        S = np.cov(frame.to_numpy(dtype=float), rowvar=False, ddof=1)
        n_obs = len(frame)
    M = A.to_numpy()
    ix, iy = nodes.index(x), nodes.index(y)
    parent_sets: List[List[int]] = []
    if method == "local":
        pa = [i for i in range(len(nodes)) if M[i, ix] and not M[ix, i]]
        sib = [i for i in range(len(nodes)) if M[i, ix] and M[ix, i]]

        def adj(a: int, b: int) -> bool:
            return bool(M[a, b] or M[b, a])

        for r in range(len(sib) + 1):
            for sub in combinations(sib, r):
                ok = all(adj(s, q) for s in sub for q in [*pa, *sub] if q != s)
                if ok:
                    parent_sets.append(sorted([*pa, *sub]))
    else:
        und = [
            (i, j)
            for i in range(len(nodes))
            for j in range(i + 1, len(nodes))
            if M[i, j] and M[j, i]
        ]
        if len(und) > MAX_UNDIRECTED_GLOBAL:
            raise MethodIncompatibility(
                f"ida: {len(und)} undirected edges; method='global' enumerates "
                f"2^{len(und)} orientations and stops at "
                f"{MAX_UNDIRECTED_GLOBAL} edges.",
                recovery_hint="Use method='local'.",
            )
        base = _vstructures(M)
        for bits in product((0, 1), repeat=len(und)):
            D = M.copy()
            for (i, j), b in zip(und, bits):
                if b:
                    D[j, i] = 0
                else:
                    D[i, j] = 0
            if _acyclic(D) and _vstructures(D) == base:
                parent_sets.append(sorted(np.flatnonzero(D[:, ix]).tolist()))
        if not parent_sets:
            raise MethodIncompatibility(
                "ida: the graph has no consistent DAG extension (not a CPDAG).",
                recovery_hint="Pass a CPDAG from sp.pc_algorithm or sp.ges.",
            )
    effects = [_ols_coef(S, ix, iy, ps) for ps in parent_sets]
    return IDAResult(
        x=x,
        y=y,
        effects=effects,
        parent_sets=[[nodes[i] for i in ps] for ps in parent_sets],
        method=method,
        detail={
            "n_obs": n_obs,
            "covariance": pd.DataFrame(S, index=nodes, columns=nodes),
        },
    )
