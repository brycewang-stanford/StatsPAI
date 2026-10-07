"""Local and hybrid structure learning: MMPC, HITON-PC and MMHC.

Constraint-based search finds, for each variable, the set of its parents and
children: the variables that stay associated with it whatever subset of the
others is conditioned on. Doing this one variable at a time needs only small
conditioning sets, so it scales to many variables. The hybrid algorithm then
runs a score-based search restricted to those candidate edges, which is both
faster and, in the simulations of the original paper, more accurate than
either kind of search alone.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import DataInsufficient, MethodIncompatibility
from .hill_climb import _is_discrete, hill_climb

__all__ = ["mmpc", "mmhc"]

Test = Callable[[int, int, Tuple[int, ...]], Tuple[float, float]]


def _gaussian_test(X: np.ndarray) -> Test:
    """Exact t test of a zero partial correlation."""
    n = X.shape[0]
    R = np.corrcoef(X, rowvar=False)

    def test(i: int, j: int, cond: Tuple[int, ...]) -> Tuple[float, float]:
        df = n - 2 - len(cond)
        if df < 1:
            return 1.0, 0.0
        if cond:
            idx = [i, j, *cond]
            try:
                P = np.linalg.inv(R[np.ix_(idx, idx)])
            except np.linalg.LinAlgError:
                return 1.0, 0.0
            r = -P[0, 1] / math.sqrt(abs(P[0, 0] * P[1, 1]))
        else:
            r = R[i, j]
        r = float(np.clip(r, -1 + 1e-15, 1 - 1e-15))
        t = r * math.sqrt(df / (1.0 - r * r))
        return float(2.0 * stats.t.sf(abs(t), df)), abs(t)

    return test


def _discrete_test(codes: np.ndarray, levels: np.ndarray) -> Test:
    """Likelihood-ratio (G-squared, mutual information) test."""
    n = codes.shape[0]

    def test(i: int, j: int, cond: Tuple[int, ...]) -> Tuple[float, float]:
        ri, rj = int(levels[i]), int(levels[j])
        config = np.zeros(n, dtype=np.int64)
        q = 1
        for c in cond:
            config = config * int(levels[c]) + codes[:, c]
            q *= int(levels[c])
        if cond:
            _, config = np.unique(config, return_inverse=True)
        counts = np.zeros((int(config.max()) + 1, ri, rj))
        np.add.at(counts, (config, codes[:, i], codes[:, j]), 1.0)
        ni = counts.sum(axis=2, keepdims=True)
        nj = counts.sum(axis=1, keepdims=True)
        nk = counts.sum(axis=(1, 2), keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            g2 = (
                2.0
                * np.where(
                    counts > 0, counts * np.log(counts * nk / (ni * nj)), 0.0
                ).sum()
            )
        df = (ri - 1) * (rj - 1) * q
        g2 = max(float(g2), 0.0)
        return float(stats.chi2.sf(g2, df)), g2 / df

    return test


def _subsets(items: Sequence[int], max_size: Optional[int]) -> Any:
    top = len(items) if max_size is None else min(len(items), max_size)
    for size in range(top + 1):
        yield from itertools.combinations(items, size)


def _max_min(
    target: int, d: int, test: Test, alpha: float, max_cond: Optional[int],
    counter: List[int],
) -> List[int]:  # fmt: skip
    """Forward phase of MMPC: add the variable whose weakest association
    with the target, over the subsets of those already chosen, is strongest."""
    weakest: Dict[int, Tuple[float, float]] = {}
    for x in range(d):
        if x != target:
            weakest[x] = test(target, x, ())
            counter[0] += 1
    cpc: List[int] = []
    while True:
        weakest = {x: v for x, v in weakest.items() if v[0] <= alpha}
        if not weakest:
            break
        best = min(weakest, key=lambda x: (weakest[x][0], -weakest[x][1], x))
        cpc.append(best)
        del weakest[best]
        others = [c for c in cpc if c != best]
        room = None if max_cond is None else max_cond - 1
        for x in list(weakest):
            for sub in _subsets(others, room):
                if max_cond is not None and len(sub) + 1 > max_cond:
                    continue
                p, s = test(target, x, (*sub, best))
                counter[0] += 1
                if (p, -s) > (weakest[x][0], -weakest[x][1]):
                    weakest[x] = (p, s)
                if p > alpha:
                    break
    return cpc


def _interleaved(
    target: int, d: int, test: Test, alpha: float, max_cond: Optional[int],
    counter: List[int],
) -> List[int]:  # fmt: skip
    """Forward phase of semi-interleaved HITON-PC: admit candidates in order
    of marginal association, dropping each newcomer some subset of the
    current set separates from the target."""
    marginal = {}
    for x in range(d):
        if x != target:
            marginal[x] = test(target, x, ())
            counter[0] += 1
    order = sorted(
        (x for x in marginal if marginal[x][0] <= alpha),
        key=lambda x: (marginal[x][0], -marginal[x][1], x),
    )
    tpc: List[int] = []
    for x in order:
        keep = True
        for sub in _subsets(tpc, max_cond):
            if not sub:
                continue
            p, _ = test(target, x, sub)
            counter[0] += 1
            if p > alpha:
                keep = False
                break
        if keep:
            tpc.append(x)
    return tpc


def _prune(
    target: int, cpc: List[int], test: Test, alpha: float,
    max_cond: Optional[int], counter: List[int],
) -> List[int]:  # fmt: skip
    """Backward phase: drop a member some subset of the others separates."""
    kept = list(cpc)
    for x in list(cpc):
        others = [c for c in kept if c != x]
        for sub in _subsets(others, max_cond):
            p, _ = test(target, x, sub)
            counter[0] += 1
            if p > alpha:
                kept.remove(x)
                break
    return kept


def mmpc(
    data: pd.DataFrame,
    variables: Optional[Sequence[str]] = None,
    *,
    method: str = "mmpc",
    data_type: str = "auto",
    alpha: float = 0.05,
    max_cond: Optional[int] = None,
) -> Dict[str, Any]:
    """Parents and children of every variable, by local constraint-based search.

    Parameters
    ----------
    data : pandas.DataFrame
    variables : sequence of str, optional
        Columns to use; all of them by default.
    method : {'mmpc', 'hiton'}, default 'mmpc'
        ``'mmpc'`` is max-min parents and children: at each step admit the
        candidate whose weakest association with the target, over all
        subsets of the variables already admitted, is strongest.
        ``'hiton'`` is semi-interleaved HITON-PC: admit candidates in order
        of their marginal association, and drop a newcomer at once if some
        subset of the current set separates it from the target. Both end
        with a pass that removes any member a subset of the others
        separates, and both keep an edge only if each end lists the other.
    data_type : {'auto', 'discrete', 'gaussian'}, default 'auto'
        As in :func:`statspai.hill_climb`. Categorical data are tested with
        the likelihood-ratio (mutual information) chi-square test,
        continuous data with the exact t test of a partial correlation.
    alpha : float, default 0.05
        Level of the conditional independence tests.
    max_cond : int, optional
        Largest conditioning set. Unlimited by default; a cap bounds the
        cost when some variable has many neighbours, and keeps the tables
        of a discrete test from thinning out.

    Returns
    -------
    dict
        ``skeleton`` (symmetric 0/1 DataFrame), ``edges`` (list of
        unordered pairs), ``neighbours`` (dict: variable -> list),
        ``asymmetric`` (pairs one end listed and the other did not, which
        are dropped), ``n_tests``, ``n_obs``, ``data_type``, ``method``,
        ``alpha``, ``variables``.

    Notes
    -----
    The result is an undirected skeleton: which variables are adjacent,
    not which way the arrows point. Under faithfulness and with no hidden
    common causes it converges to the skeleton of the true graph. In a
    finite sample each test can err; a non-empty ``asymmetric`` list is a
    sign that some did.

    On 24 simulated data sets (five to eight variables, Gaussian and
    categorical, 500 to 3,000 rows) both methods return the skeleton of
    ``bnlearn::mmpc`` and ``si.hiton.pc`` every time. That is not
    guaranteed in general: the algorithms are the published ones, written
    independently, and which subsets get tested depends on the order in
    which candidates are admitted and ties are broken.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> a = rng.normal(size=n)
    >>> b = a + rng.normal(size=n)
    >>> c = b + rng.normal(size=n)                 # a -> b -> c
    >>> out = sp.mmpc(pd.DataFrame({"a": a, "b": b, "c": c}))
    >>> out["edges"]
    [('a', 'b'), ('b', 'c')]

    References
    ----------
    [@tsamardinos2006maxmin], [@aliferis2010local]
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility("mmpc: data must be a DataFrame.")
    how = str(method).lower().replace("si.hiton.pc", "hiton").replace("_pc", "")
    if how not in ("mmpc", "hiton"):
        raise MethodIncompatibility(
            f"mmpc: method must be 'mmpc' or 'hiton', got {method!r}."
        )
    if not (0.0 < alpha < 1.0):
        raise MethodIncompatibility("mmpc: alpha must be between 0 and 1.")
    if max_cond is not None and max_cond < 0:
        raise MethodIncompatibility("mmpc: max_cond cannot be negative.")
    cols = list(variables) if variables is not None else list(data.columns)
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise MethodIncompatibility(f"mmpc: columns not in data: {missing}.")
    if len(cols) < 2:
        raise MethodIncompatibility("mmpc: at least two variables are needed.")
    frame = data[cols].dropna()
    if len(frame) < 10:
        raise DataInsufficient("mmpc: fewer than 10 complete rows.")
    kind = str(data_type).lower()
    if kind not in ("auto", "discrete", "gaussian"):
        raise MethodIncompatibility(
            "mmpc: data_type must be 'auto', 'discrete' or 'gaussian', "
            f"got {data_type!r}."
        )
    if kind == "auto":
        flags = [_is_discrete(frame[c]) for c in cols]
        if all(flags):
            kind = "discrete"
        elif not any(flags):
            kind = "gaussian"
        else:
            raise MethodIncompatibility(
                "mmpc: the columns mix categorical and numeric types "
                f"(categorical: {[c for c, f in zip(cols, flags) if f]}).",
                recovery_hint="Pass data_type='discrete' or 'gaussian'.",
            )
    if kind == "gaussian":
        bad = [c for c in cols if not pd.api.types.is_numeric_dtype(frame[c])]
        if bad:
            raise MethodIncompatibility(
                f"mmpc: data_type='gaussian' needs numeric columns; {bad} are not."
            )
        X = frame.to_numpy(dtype=float)
        if np.any(X.std(axis=0) == 0):
            raise DataInsufficient("mmpc: a column is constant.")
        test = _gaussian_test(X)
    else:
        cats = [pd.Categorical(frame[c]) for c in cols]
        codes = np.column_stack([c.codes for c in cats]).astype(np.int64)
        levels = np.array([len(c.categories) for c in cats])
        if np.any(levels < 2):
            raise DataInsufficient("mmpc: a column has a single level.")
        test = _discrete_test(codes, levels)

    cache: Dict[Tuple[int, int, Tuple[int, ...]], Tuple[float, float]] = {}

    def cached(i: int, j: int, cond: Tuple[int, ...]) -> Tuple[float, float]:
        key = (min(i, j), max(i, j), tuple(sorted(cond)))
        if key not in cache:
            cache[key] = test(key[0], key[1], key[2])
        return cache[key]

    d = len(cols)
    counter = [0]
    forward = _max_min if how == "mmpc" else _interleaved
    found: List[Set[int]] = []
    for t in range(d):
        cpc = forward(t, d, cached, alpha, max_cond, counter)
        found.append(set(_prune(t, cpc, cached, alpha, max_cond, counter)))
    adj = np.zeros((d, d), dtype=int)
    asymmetric: List[Tuple[str, str]] = []
    for i in range(d):
        for j in range(i + 1, d):
            if j in found[i] and i in found[j]:
                adj[i, j] = adj[j, i] = 1
            elif j in found[i] or i in found[j]:
                asymmetric.append((cols[i], cols[j]))
    edges = [(cols[i], cols[j]) for i in range(d) for j in range(i + 1, d)
             if adj[i, j]]  # fmt: skip
    return {
        "skeleton": pd.DataFrame(adj, index=cols, columns=cols),
        "edges": edges,
        "neighbours": {cols[i]: [cols[j] for j in range(d) if adj[i, j]]
                       for i in range(d)},  # fmt: skip
        "asymmetric": asymmetric,
        "n_tests": len(cache),
        "n_obs": int(len(frame)),
        "data_type": kind,
        "method": how,
        "alpha": float(alpha),
        "variables": cols,
    }


def mmhc(
    data: pd.DataFrame,
    variables: Optional[Sequence[str]] = None,
    *,
    restrict: str = "mmpc",
    data_type: str = "auto",
    alpha: float = 0.05,
    max_cond: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Max-min hill climbing: a score search over a tested skeleton.

    Parameters
    ----------
    data : pandas.DataFrame
    variables : sequence of str, optional
    restrict : {'mmpc', 'hiton'}, default 'mmpc'
        The local search that proposes the candidate edges, see
        :func:`statspai.mmpc`.
    data_type : {'auto', 'discrete', 'gaussian'}, default 'auto'
    alpha : float, default 0.05
        Level of the tests in the restriction step.
    max_cond : int, optional
        Largest conditioning set in the restriction step.
    **kwargs
        Passed to :func:`statspai.hill_climb` (``max_parents=``,
        ``forbidden=``, ``required=``, ``restarts=``, ``seed=``).

    Returns
    -------
    dict
        What :func:`statspai.hill_climb` returns, plus ``skeleton`` and
        ``candidate_edges`` (the pairs the search was allowed to connect)
        and ``restrict``.

    Notes
    -----
    Two steps. First, :func:`statspai.mmpc` finds the pairs of variables no
    conditioning set separates. Second, greedy hill climbing on the BIC
    adds, removes and reverses arcs among those pairs only. The first step
    cuts the search space from every pair to a sparse set, and removes the
    spurious arcs a pure score search adds between variables that are
    merely correlated through others.

    The restriction can only remove candidate arcs, so the score of the
    result is never above what an unrestricted search could reach, and a
    true edge the tests miss cannot be recovered. With little data, where
    the tests have low power, the plain :func:`statspai.hill_climb` may
    find more of the graph.

    The original algorithm scores with BDeu and a tabu list; this follows
    ``bnlearn::mmhc`` in scoring with the BIC and plain hill climbing. On
    the 24 data sets of the comparison the score was never below
    bnlearn's: equal on 18 and higher on 6, where the search here leaves a
    plateau bnlearn's stops on.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> a = rng.normal(size=n)
    >>> b = rng.normal(size=n)
    >>> c = a + b + 0.5 * rng.normal(size=n)       # a -> c <- b
    >>> out = sp.mmhc(pd.DataFrame({"a": a, "b": b, "c": c}))
    >>> sorted(out["edges"])
    [('a', 'c'), ('b', 'c')]

    References
    ----------
    [@tsamardinos2006maxmin]
    """
    local = mmpc(
        data, variables, method=restrict, data_type=data_type, alpha=alpha,
        max_cond=max_cond,
    )  # fmt: skip
    cols = local["variables"]
    allowed = {frozenset(e) for e in local["edges"]}
    forbidden = [tuple(pair) for pair in kwargs.pop("forbidden", None) or []]
    for a in cols:
        for b in cols:
            if a != b and frozenset((a, b)) not in allowed:
                forbidden.append((a, b))
    for a, b in kwargs.get("required", None) or []:
        if frozenset((a, b)) not in allowed:
            raise MethodIncompatibility(
                f"mmhc: the required arc ({a!r}, {b!r}) joins two variables "
                "the tests found separable.",
                recovery_hint="Use sp.hill_climb(required=...) to impose it "
                "without the restriction.",
            )
    out = hill_climb(
        data, cols, data_type=local["data_type"], forbidden=forbidden, **kwargs
    )
    out["skeleton"] = local["skeleton"]
    out["candidate_edges"] = local["edges"]
    out["restrict"] = local["method"]
    return out
