"""The parts of FCI that come after the PC skeleton.

* :func:`possible_dsep_removal` -- the second adjacency pass. With latent
  variables two non-adjacent nodes need not be separated by a subset of
  either one's neighbours; a separating set does exist inside
  Possible-D-SEP, the nodes reachable along paths on which every
  consecutive triple is a collider or a triangle.
* :func:`orient_pag` -- colliders on unshielded triples, then the ten
  orientation rules of Zhang (2008), which are complete for ancestral
  graphs with latent and selection variables.

The graph is one integer matrix ``P``: ``P[i, j]`` is the mark at the ``j``
end of the edge between ``i`` and ``j`` (0 none, 1 circle, 2 arrowhead,
3 tail), the layout of ``pcalg``'s ``amat`` for a PAG.

References
----------
[@zhang2008completeness], [@colombo2012learning]
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

NONE, CIRCLE, ARROW, TAIL = 0, 1, 2, 3

SepSets = Dict[Tuple[int, int], Set[int]]


def circle_graph(adj: np.ndarray) -> np.ndarray:
    """Every edge of the skeleton as ``o-o``."""
    P = np.where(adj != 0, CIRCLE, NONE).astype(int)
    np.fill_diagonal(P, NONE)
    return P


def orient_colliders(P: np.ndarray, sep_sets: SepSets) -> None:
    """``a *-> b <-* c`` for every unshielded triple whose separating set
    leaves ``b`` out. In a PAG both ends of an edge may carry an arrowhead,
    so two colliders never contradict each other."""
    d = P.shape[0]
    for b in range(d):
        nbrs = [int(v) for v in np.flatnonzero(P[b])]
        for a, c in combinations(nbrs, 2):
            if P[a, c] != NONE:
                continue
            sep = sep_sets.get((a, c), sep_sets.get((c, a)))
            if sep is not None and b not in sep:
                P[a, b] = ARROW
                P[c, b] = ARROW


def possible_dsep(P: np.ndarray, x: int) -> List[int]:
    """Nodes reachable from ``x`` along paths on which every consecutive
    triple ``u, v, w`` is a collider at ``v`` or has ``u`` and ``w``
    adjacent."""
    reached: Set[int] = set()
    seen: Set[Tuple[int, int]] = set()
    frontier = [(x, int(v)) for v in np.flatnonzero(P[x])]
    seen.update(frontier)
    while frontier:
        u, v = frontier.pop()
        reached.add(v)
        for w in np.flatnonzero(P[v]):
            w = int(w)
            if w == u or w == x or (v, w) in seen:
                continue
            collider = P[u, v] == ARROW and P[w, v] == ARROW
            triangle = P[u, w] != NONE
            if collider or triangle:
                seen.add((v, w))
                frontier.append((v, w))
    return sorted(reached)


def possible_dsep_removal(
    adj: np.ndarray,
    sep_sets: SepSets,
    alpha: float,
    pvalue: Callable[[int, int, List[int]], float],
    max_cond_size: Optional[int] = None,
) -> int:
    """Remove edges separated by a subset of Possible-D-SEP.

    Possible-D-SEP is computed once, on the skeleton with its colliders
    oriented. For an edge ``x - y`` only conditioning sets that reach outside
    the neighbours of ``x`` are tried: those inside were tried by the
    skeleton search. ``adj`` and ``sep_sets`` are updated in place; returns
    the number of edges removed.
    """
    d = adj.shape[0]
    P = circle_graph(adj)
    orient_colliders(P, sep_sets)
    pds = [possible_dsep(P, x) for x in range(d)]
    limit = d if max_cond_size is None else int(max_cond_size)
    removed = 0
    for x in range(d):
        for y in [int(v) for v in np.flatnonzero(adj[x])]:
            if adj[x, y] == 0:
                continue
            candidates = [v for v in pds[x] if v != y]
            nbrs = {int(v) for v in np.flatnonzero(adj[x])} - {y}
            outside = [v for v in candidates if v not in nbrs]
            if not outside:
                continue
            found = False
            for size in range(1, min(len(candidates), limit) + 1):
                for S in combinations(candidates, size):
                    if set(S) <= nbrs:
                        continue
                    if pvalue(x, y, list(S)) >= alpha:
                        adj[x, y] = adj[y, x] = 0
                        sep_sets[(x, y)] = set(S)
                        sep_sets[(y, x)] = set(S)
                        removed += 1
                        found = True
                        break
                if found:
                    break
    return removed


# ----------------------------------------------------------------- path search
def _uncovered_paths(
    P: np.ndarray,
    start: int,
    second: int,
    end: int,
    step_ok: Callable[[int, int], bool],
) -> Iterator[List[int]]:
    """Uncovered paths ``start, second, ..., end`` whose every edge passes
    ``step_ok(u, v)``; consecutive triples have non-adjacent outer nodes."""
    if not step_ok(start, second):
        return
    stack: List[List[int]] = [[start, second]]
    while stack:
        path = stack.pop()
        u, v = path[-2], path[-1]
        if v == end:
            yield path
            continue
        for w in np.flatnonzero(P[v]):
            w = int(w)
            if w in path or P[u, w] != NONE or not step_ok(v, w):
                continue
            stack.append(path + [w])


def _pd_step(P: np.ndarray) -> Callable[[int, int], bool]:
    """Edge ``u - v`` usable on a potentially directed path from u to v."""
    return lambda u, v: bool(P[v, u] != ARROW and P[u, v] != TAIL)


def _circle_step(P: np.ndarray) -> Callable[[int, int], bool]:
    return lambda u, v: bool(P[u, v] == CIRCLE and P[v, u] == CIRCLE)


def _discriminating_path_start(P: np.ndarray, a: int, b: int, c: int) -> Optional[int]:
    """The far end ``d`` of a discriminating path ``d, ..., a, b, c`` for
    ``b``, or None. Every node between ``d`` and ``b`` is a collider on the
    path and a parent of ``c``; ``d`` is not adjacent to ``c``."""
    # paths are grown backwards from a; each state is (previous, current)
    # with `current` nearer to d
    visited: Set[int] = {a, b, c}
    frontier: List[Tuple[int, int]] = [(b, a)]
    while frontier:
        nxt, cur = frontier.pop(0)
        for v in np.flatnonzero(P[cur]):
            v = int(v)
            if v in visited or P[v, cur] != ARROW:
                continue
            # cur must be a collider between v and nxt: arrowheads at cur
            # from both sides (the one from nxt was checked by the caller
            # or on the previous step)
            if P[v, c] == NONE:
                return v
            # v stays on the path only as a collider that is a parent of c
            if P[cur, v] == ARROW and P[v, c] == ARROW and P[c, v] == TAIL:
                visited.add(v)
                frontier.append((cur, v))
    return None


# ----------------------------------------------------------------------- rules
def apply_rules(P: np.ndarray, sep_sets: SepSets, max_sweeps: int = 1000) -> None:
    """Zhang's orientation rules R1-R10, to a fixed point."""
    d = P.shape[0]

    def adjacent(i: int, j: int) -> bool:
        return bool(P[i, j] != NONE)

    for _ in range(max_sweeps):
        before = P.copy()

        # R1: a *-> b o-* c, a and c non-adjacent  =>  b -> c
        for b in range(d):
            for a in np.flatnonzero(P[:, b] == ARROW):
                for c in np.flatnonzero(P[:, b] == CIRCLE):
                    a, c = int(a), int(c)
                    if a != c and not adjacent(a, c) and P[c, b] == CIRCLE:
                        P[c, b] = TAIL
                        P[b, c] = ARROW

        # R2: a -> b *-> c or a *-> b -> c, with a *-o c  =>  a *-> c
        for a in range(d):
            for c in np.flatnonzero(P[a] == CIRCLE):
                c = int(c)
                for b in range(d):
                    if b in (a, c) or P[a, b] != ARROW or P[b, c] != ARROW:
                        continue
                    if P[b, a] == TAIL or P[c, b] == TAIL:
                        P[a, c] = ARROW
                        break

        # R3: a *-> b <-* c, a *-o t o-* c, a and c non-adjacent, t *-o b
        #     =>  t *-> b
        for b in range(d):
            heads = [int(v) for v in np.flatnonzero(P[:, b] == ARROW)]
            for t in np.flatnonzero(P[:, b] == CIRCLE):
                t = int(t)
                if P[t, b] != CIRCLE:
                    continue
                for a, c in combinations(heads, 2):
                    if not adjacent(a, c) and P[a, t] == CIRCLE and P[c, t] == CIRCLE:
                        P[t, b] = ARROW
                        break

        # R4: discriminating path d, ..., a, b, c for b, with b o-* c
        for b in range(d):
            for c in np.flatnonzero(P[:, b] == CIRCLE):
                c = int(c)
                if P[c, b] != CIRCLE:
                    continue
                for a in range(d):
                    if a in (b, c) or P[c, b] != CIRCLE:
                        continue
                    # a <-* b, a -> c
                    if P[b, a] != ARROW or P[a, c] != ARROW or P[c, a] != TAIL:
                        continue
                    far = _discriminating_path_start(P, a, b, c)
                    if far is None:
                        continue
                    sep = sep_sets.get((far, c), sep_sets.get((c, far), set()))
                    if b in sep:
                        P[b, c] = ARROW
                        P[c, b] = TAIL
                    else:
                        P[a, b] = ARROW
                        P[b, c] = ARROW
                        P[c, b] = ARROW
                    break

        # R5: a o-o b and an uncovered circle path a, c, ..., t, b with
        #     a, t non-adjacent and b, c non-adjacent  =>  all of it undirected
        for a in range(d):
            for b in np.flatnonzero((P[a] == CIRCLE) & (P[:, a] == CIRCLE)):
                b = int(b)
                if not (P[a, b] == CIRCLE and P[b, a] == CIRCLE):
                    continue
                done = False
                for c in np.flatnonzero(P[a]):
                    c = int(c)
                    if c == b or adjacent(c, b):
                        continue
                    for path in _uncovered_paths(P, a, c, b, _circle_step(P)):
                        t = path[-2]
                        if len(path) < 4 or adjacent(a, t):
                            continue
                        P[a, b] = P[b, a] = TAIL
                        for u, v in zip(path[:-1], path[1:]):
                            P[u, v] = P[v, u] = TAIL
                        done = True
                        break
                    if done:
                        break

        # R6: a - b o-* c  =>  b -* c
        for b in range(d):
            if any(P[a, b] == TAIL and P[b, a] == TAIL for a in range(d)):
                for c in np.flatnonzero(P[:, b] == CIRCLE):
                    P[int(c), b] = TAIL

        # R7: a -o b o-* c, a and c non-adjacent  =>  b -* c
        for b in range(d):
            for a in range(d):
                if not (P[b, a] == TAIL and P[a, b] == CIRCLE):
                    continue
                for c in np.flatnonzero(P[:, b] == CIRCLE):
                    c = int(c)
                    if c != a and not adjacent(a, c):
                        P[c, b] = TAIL

        # R8: a -> b -> c or a -o b -> c, with a o-> c  =>  a -> c
        for a in range(d):
            for c in np.flatnonzero(P[a] == ARROW):
                c = int(c)
                if P[c, a] != CIRCLE:
                    continue
                for b in range(d):
                    if b in (a, c):
                        continue
                    first = P[b, a] == TAIL and P[a, b] in (ARROW, CIRCLE)
                    second = P[b, c] == ARROW and P[c, b] == TAIL
                    if first and second:
                        P[c, a] = TAIL
                        break

        # R9: a o-> c and an uncovered potentially directed path
        #     a, b, ..., c with b, c non-adjacent  =>  a -> c
        for a in range(d):
            for c in np.flatnonzero(P[a] == ARROW):
                c = int(c)
                if P[c, a] != CIRCLE:
                    continue
                done = False
                for b in np.flatnonzero(P[a]):
                    b = int(b)
                    if b == c or adjacent(b, c):
                        continue
                    for _path in _uncovered_paths(P, a, b, c, _pd_step(P)):
                        P[c, a] = TAIL
                        done = True
                        break
                    if done:
                        break

        # R10: a o-> c, b -> c <- t, uncovered potentially directed paths
        #      from a to b and from a to t whose first steps m, w are
        #      distinct and non-adjacent  =>  a -> c
        for a in range(d):
            for c in np.flatnonzero(P[a] == ARROW):
                c = int(c)
                if P[c, a] != CIRCLE:
                    continue
                parents = [
                    int(v)
                    for v in np.flatnonzero((P[:, c] == ARROW) & (P[c] == TAIL))
                    if v != a
                ]
                done = False
                for b, t in combinations(parents, 2):
                    firsts_b = _first_steps(P, a, b)
                    firsts_t = _first_steps(P, a, t)
                    if any(
                        m != w and not adjacent(m, w)
                        for m in firsts_b
                        for w in firsts_t
                    ):
                        P[c, a] = TAIL
                        done = True
                        break
                if done:
                    continue

        if np.array_equal(before, P):
            return


def _first_steps(P: np.ndarray, a: int, target: int) -> Set[int]:
    """First nodes after ``a`` on uncovered potentially directed paths from
    ``a`` to ``target``."""
    out: Set[int] = set()
    for m in np.flatnonzero(P[a]):
        m = int(m)
        if m == target:
            if _pd_step(P)(a, m):
                out.add(m)
            continue
        for _path in _uncovered_paths(P, a, m, target, _pd_step(P)):
            out.add(m)
            break
    return out


def orient_pag(adj: np.ndarray, sep_sets: SepSets) -> np.ndarray:
    """Circle graph on the skeleton, colliders, then the ten rules."""
    P = circle_graph(adj)
    orient_colliders(P, sep_sets)
    apply_rules(P, sep_sets)
    return P
