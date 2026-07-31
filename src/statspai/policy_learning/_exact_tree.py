"""Exact welfare-maximising policy-tree search.

The policy-learning objective (Athey & Wager 2021) is

    max_{pi in Pi_d}  sum_i Gamma_i * pi(X_i)

over the class ``Pi_d`` of axis-aligned decision trees of depth ``<= d``,
where ``Gamma_i`` is the doubly-robust (AIPW) score for *treating* unit
``i`` and ``pi(X_i) in {0, 1}``.  Writing the two-action reward matrix as
``(0, Gamma_i)`` makes this identical to the objective solved by
``policytree::policy_tree`` (Sverdrup, Kanodia, Zhou, Athey & Wager 2020),
so the two implementations are directly comparable.

Why this module exists
----------------------
The previous in-house search was **greedy**: the root split was scored as
if both children were terminal leaves, and only then did the routine
recurse.  For ``depth = 2`` that is not the welfare optimum -- the root
split that looks best when its children are leaves is frequently *not*
the root split that admits the best pair of depth-1 subtrees.  The search
also subsampled candidate thresholds to at most 50 quantiles per feature,
so it did not even return the greedy optimum over the full split grid.

This module implements the exact optimum for ``depth <= 2`` over the
complete grid of distinct covariate values, matching the ``<=``-goes-left
split convention and the "smallest permitted terminal node" reading of
``min_leaf_size`` used by ``policytree``.

Algorithm
---------
Let ``S`` be a node's index set.  For every feature ``k`` maintain the
rank-indexed partial sums

    SA[k, r] = sum_{i in S, rank(x_ik) <= r} Gamma_i
    SC[k, r] = #{i in S : rank(x_ik) <= r}

so that the exact depth-1 value of ``S`` is a single vectorised reduction

    max_{k, r} [ max(SA[k, r], 0) + max(totA - SA[k, r], 0) ]

subject to both terminal nodes holding at least ``min_leaf_size`` units.

For depth 2 we sweep the root threshold along each feature in rank order.
Each step moves one group of tied points from the right child to the left
child, updating ``SA`` / ``SC`` incrementally, so both children's exact
depth-1 values are recomputed with two ``cumsum`` calls rather than a
fresh scan.  Complexity is ``O(p^2 n^2)`` element-operations -- the
intrinsic cost of exact depth-2 tree search, and the same order as the
reference C++ implementation -- but the inner loop is fully vectorised.

References
----------
[@athey2021policy], [@zhou2023offline]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

TreeNode = Dict[str, Any]

_NEG_INF = -np.inf

# Element-operation budget for the exact depth-2 sweep under
# ``search="auto"``.  The sweep costs about ``p^2 * n * U`` float
# operations (U = distinct values per feature, <= n).  5e8 keeps a
# five-covariate problem exact up to roughly n = 10,000 while stopping
# ``sp.policy_tree`` from silently turning a 100k-row call into a
# multi-minute exhaustive search.
_AUTO_EXACT_BUDGET = 5e8


def _leaf(scores: np.ndarray) -> TreeNode:
    """Terminal node: treat iff the node's mean AIPW score is positive."""
    n = int(scores.size)
    mean = float(np.mean(scores)) if n else 0.0
    return {"type": "leaf", "action": 1 if mean > 0 else 0, "value": mean, "n": n}


class _RankIndex:
    """Per-feature distinct values, ranks, and the candidate-split mask.

    ``ranks[i, k]`` is the position of ``X[i, k]`` among the sorted
    distinct values of column ``k``.  A split at rank ``r`` of feature
    ``k`` sends ``{i : rank(x_ik) <= r}`` left, mirroring the
    ``x <= threshold`` convention of ``policytree`` and of
    :meth:`PolicyTree._predict_tree`.
    """

    def __init__(self, X: np.ndarray, split_step: int = 1) -> None:
        n, p = X.shape
        uvals: List[np.ndarray] = []
        ranks = np.empty((n, p), dtype=np.intp)
        for k in range(p):
            u, inv = np.unique(X[:, k], return_inverse=True)
            uvals.append(u)
            ranks[:, k] = inv.reshape(-1)
        self.n = n
        self.p = p
        self.uvals = uvals
        self.ranks = ranks
        self.U = max((len(u) for u in uvals), default=1)

        # Candidate mask over the padded (p, U) grid.  Rank r is a usable
        # split only if a strictly larger value exists (r <= len(u) - 2);
        # padded columns beyond a feature's distinct-value count are never
        # candidates.  ``split_step`` thins the grid the same way
        # ``policytree``'s ``split.step`` does.
        cand = np.zeros((p, self.U), dtype=bool)
        for k in range(p):
            n_splits = len(uvals[k]) - 1
            if n_splits > 0:
                cand[k, np.arange(0, n_splits, split_step)] = True
        self.cand = cand

    def accumulate(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rank-indexed score sums / counts over every observation."""
        A = np.zeros((self.p, self.U), dtype=np.float64)
        C = np.zeros((self.p, self.U), dtype=np.int64)
        for k in range(self.p):
            A[k] = np.bincount(self.ranks[:, k], weights=scores, minlength=self.U)
            C[k] = np.bincount(self.ranks[:, k], minlength=self.U)
        return A, C


def _depth1_best(
    SA: np.ndarray,
    SC: np.ndarray,
    tot_a: float,
    tot_c: int,
    cand: np.ndarray,
    min_leaf: int,
) -> Tuple[float, Optional[Tuple[int, int]]]:
    """Exact depth-1 value of a node, given its rank-indexed partial sums.

    Returns ``(value, split)`` where ``split`` is ``(feature, rank)`` or
    ``None`` when the node is best left (or forced to remain) terminal.
    ``value`` is ``-inf`` when the node cannot host any admissible tree,
    i.e. it holds fewer than ``min_leaf`` observations.
    """
    if tot_c < min_leaf:
        return _NEG_INF, None

    leaf_value = max(tot_a, 0.0)

    # A split is admissible only when *both* terminal nodes are large
    # enough.  Padded grid positions satisfy SC == tot_c, hence
    # tot_c - SC == 0 < min_leaf, and drop out automatically.
    ok = cand & (SC >= min_leaf) & ((tot_c - SC) >= min_leaf)
    if not ok.any():
        return leaf_value, None

    value = np.maximum(SA, 0.0) + np.maximum(tot_a - SA, 0.0)
    value = np.where(ok, value, _NEG_INF)
    flat = int(np.argmax(value))
    best = float(value.flat[flat])
    if best <= leaf_value:
        # A split never scores below the terminal option, so equality
        # means the split is redundant; prefer the simpler tree.
        return leaf_value, None
    k, r = np.unravel_index(flat, value.shape)
    return best, (int(k), int(r))


def _materialise_depth1(
    X: np.ndarray,
    scores: np.ndarray,
    index: _RankIndex,
    rows: np.ndarray,
    split: Optional[Tuple[int, int]],
) -> TreeNode:
    """Turn a ``(feature, rank)`` decision into the nested node dict."""
    if split is None:
        return _leaf(scores[rows])
    k, r = split
    threshold = float(index.uvals[k][r])
    left_mask = X[rows, k] <= threshold
    left_rows = rows[left_mask]
    right_rows = rows[~left_mask]
    return {
        "type": "split",
        "feature": k,
        "threshold": threshold,
        "left": _leaf(scores[left_rows]),
        "right": _leaf(scores[right_rows]),
        "n": int(rows.size),
    }


def exact_policy_tree(
    X: np.ndarray,
    scores: np.ndarray,
    *,
    max_depth: int,
    min_leaf_size: int,
    split_step: int = 1,
) -> TreeNode:
    """Welfare-maximising policy tree for ``max_depth <= 2``.

    Parameters
    ----------
    X : ndarray (n, p)
        Policy covariates.
    scores : ndarray (n,)
        Doubly-robust scores ``Gamma_i`` for treating unit ``i``.
    max_depth : int
        Tree depth; must be ``<= 2``.
    min_leaf_size : int
        Smallest permitted terminal node, matching ``policytree``'s
        ``min.node.size``.
    split_step : int, default 1
        Consider only every ``split_step``-th distinct value of each
        covariate.  ``1`` searches the complete grid.

    Returns
    -------
    dict
        Nested ``{"type": "split"|"leaf", ...}`` node, the same shape the
        greedy search produced.
    """
    if max_depth > 2:
        raise ValueError("exact_policy_tree supports max_depth <= 2")
    n, p = X.shape
    min_leaf = max(1, int(min_leaf_size))
    rows_all = np.arange(n, dtype=np.intp)

    if max_depth <= 0 or n < 2 * min_leaf or p == 0:
        return _leaf(scores)

    index = _RankIndex(X, split_step=split_step)
    A_tot, C_tot = index.accumulate(scores)
    SA_tot = np.cumsum(A_tot, axis=1)
    SC_tot = np.cumsum(C_tot, axis=1)
    tot_a = float(np.sum(scores))

    # Depth-1 (and, implicitly, depth-0) optimum over the whole sample.
    best_value, root_split = _depth1_best(
        SA_tot, SC_tot, tot_a, n, index.cand, min_leaf
    )
    best_tree = _materialise_depth1(X, scores, index, rows_all, root_split)
    if max_depth == 1:
        return best_tree

    # Depth-2: sweep the root threshold along each feature, keeping the
    # left child's rank-indexed partial sums up to date incrementally.
    _Split = Optional[Tuple[int, int]]
    best_root: Optional[Tuple[int, int, _Split, _Split]] = None
    for j in range(p):
        order = np.argsort(index.ranks[:, j], kind="stable")
        rank_j = index.ranks[order, j]
        # Start index of each tied group in the sorted order.
        group_start = np.flatnonzero(np.r_[True, rank_j[1:] != rank_j[:-1]])
        group_rank = rank_j[group_start]
        group_end = np.r_[group_start[1:], n]

        A = np.zeros((index.p, index.U), dtype=np.float64)
        C = np.zeros((index.p, index.U), dtype=np.int64)
        tot_a_left = 0.0
        tot_c_left = 0
        n_splits = len(index.uvals[j]) - 1

        for g, r in enumerate(group_rank):
            if r > n_splits - 1:
                break
            grp = order[group_start[g] : group_end[g]]
            grp_ranks = index.ranks[grp]  # (g_size, p)
            feat_idx = np.tile(np.arange(index.p, dtype=np.intp), grp.size)
            np.add.at(A, (feat_idx, grp_ranks.ravel()), np.repeat(scores[grp], index.p))
            np.add.at(C, (feat_idx, grp_ranks.ravel()), 1)
            tot_a_left += float(np.sum(scores[grp]))
            tot_c_left += int(grp.size)

            if not index.cand[j, r]:
                continue
            if tot_c_left < min_leaf or (n - tot_c_left) < min_leaf:
                continue

            SA = np.cumsum(A, axis=1)
            SC = np.cumsum(C, axis=1)
            left_value, left_split = _depth1_best(
                SA, SC, tot_a_left, tot_c_left, index.cand, min_leaf
            )
            right_value, right_split = _depth1_best(
                SA_tot - SA,
                SC_tot - SC,
                tot_a - tot_a_left,
                n - tot_c_left,
                index.cand,
                min_leaf,
            )
            if left_value == _NEG_INF or right_value == _NEG_INF:
                continue
            total = left_value + right_value
            if total > best_value:
                best_value = total
                best_root = (j, int(r), left_split, right_split)

    if best_root is None:
        return best_tree

    j, r, left_split, right_split = best_root
    threshold = float(index.uvals[j][r])
    left_mask = X[:, j] <= threshold
    left_rows = rows_all[left_mask]
    right_rows = rows_all[~left_mask]
    return {
        "type": "split",
        "feature": j,
        "threshold": threshold,
        "left": _materialise_depth1(X, scores, index, left_rows, left_split),
        "right": _materialise_depth1(X, scores, index, right_rows, right_split),
        "n": int(n),
    }


def exact_search_is_affordable(n: int, p: int, max_depth: int) -> bool:
    """Whether the exact depth-2 sweep fits the ``search='auto'`` budget."""
    if max_depth <= 1:
        return True
    return float(p) ** 2 * float(n) ** 2 <= _AUTO_EXACT_BUDGET


def warn_greedy_fallback(n: int, p: int, max_depth: int) -> None:
    """Loud notice that ``search='auto'`` declined the exact search."""
    warnings.warn(
        f"policy_tree: exact depth-{max_depth} search over n={n} rows and "
        f"p={p} policy covariates exceeds the automatic budget "
        f"(cost ~ p^2*n^2 = {p ** 2 * n ** 2:.3g} > {_AUTO_EXACT_BUDGET:.3g}); "
        "falling back to the greedy search, whose tree is not guaranteed to "
        "maximise the estimated policy value. Pass search='exact' to force "
        "the exhaustive search, or split_step>1 to thin the candidate grid.",
        UserWarning,
        stacklevel=3,
    )
