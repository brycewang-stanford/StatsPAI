"""Multi-action policy trees.

The objective is ``max_{pi in Pi_d} sum_i Gamma[i, pi(X_i)]`` over
axis-aligned trees of depth ``<= d`` whose leaves each pick one of ``A``
actions, with ``Gamma`` an ``n x A`` reward matrix (doubly robust scores,
one column per action). This is ``policytree::policy_tree(X, Gamma)``
(Sverdrup et al.; Zhou, Athey and Wager 2023, [@zhou2023offline]); the
binary search in :mod:`._exact_tree` is its ``A = 2`` case written on the
contrast ``Gamma_1 - Gamma_0``.

The exact depth ``<= 2`` search below is the binary one with an action
axis added to the rank-indexed partial sums: a node's value is
``max_a sum_{i in node} Gamma[i, a]``, a depth-1 split's value
``max_a SA[k, r, a] + max_a (tot[a] - SA[k, r, a])``. Splits follow the
``x <= threshold`` goes-left convention; ``min_leaf_size`` is the smallest
permitted terminal node. Ties: the first action (column order) wins in a
leaf, the first (feature, threshold) in a split search, and a split must
strictly beat the unsplit node. Values are compared up to rounding
(``_tie_tol``): candidates whose values differ only in the last bits are
exact ties computed in a different order, and letting that noise decide
made the chosen threshold depend on the platform's summation (the
multi-action policytree fixture has two exactly tied root splits).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..exceptions import DataInsufficient, MethodIncompatibility
from ._exact_tree import _RankIndex

TreeNode = Dict[str, Any]
_NEG = -np.inf


def _tie_tol(R: np.ndarray) -> float:
    """Rounding scale of a sum of rows of ``R``: values closer than this
    are treated as equal (``1e-12`` of the total absolute reward)."""
    return 1e-12 * float(np.abs(R).sum())


def _leaf(R: np.ndarray) -> TreeNode:
    sums = R.sum(axis=0)
    a = int(np.argmax(sums))
    n = int(R.shape[0])
    return {"type": "leaf", "action": a, "value": float(sums[a] / max(n, 1)), "n": n}


def _accumulate(index: _RankIndex, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = R.shape[1]
    S = np.zeros((index.p, index.U, A))
    C = np.zeros((index.p, index.U), dtype=np.int64)
    for k in range(index.p):
        for a in range(A):
            S[k, :, a] = np.bincount(
                index.ranks[:, k], weights=R[:, a], minlength=index.U
            )
        C[k] = np.bincount(index.ranks[:, k], minlength=index.U)
    return S, C


def _depth1(
    SA: np.ndarray,
    SC: np.ndarray,
    tot: np.ndarray,
    n: int,
    cand: np.ndarray,
    m: int,
    tol: float = 0.0,
) -> Tuple[float, Optional[Tuple[int, int]]]:
    if n < m:
        return _NEG, None
    leaf = float(tot.max())
    ok = cand & (SC >= m) & ((n - SC) >= m)
    if not ok.any():
        return leaf, None
    val = SA.max(axis=2) + (tot[None, None, :] - SA).max(axis=2)
    val = np.where(ok, val, _NEG)
    # first (feature, threshold) among the splits tied with the maximum
    flat = int(np.argmax(val >= val.max() - tol))
    best = float(val.flat[flat])
    if best <= leaf + tol:
        return leaf, None
    k, r = np.unravel_index(flat, val.shape)
    return best, (int(k), int(r))


def _node(X, R, index, rows, split) -> TreeNode:
    if split is None:
        return _leaf(R[rows])
    k, r = split
    thr = float(index.uvals[k][r])
    left = rows[X[rows, k] <= thr]
    right = rows[X[rows, k] > thr]
    return {
        "type": "split",
        "feature": k,
        "threshold": thr,
        "left": _leaf(R[left]),
        "right": _leaf(R[right]),
        "n": int(rows.size),
    }


def exact_multi_action_tree(
    X: np.ndarray,
    R: np.ndarray,
    *,
    max_depth: int,
    min_leaf_size: int,
    split_step: int = 1,
) -> TreeNode:
    """Reward-maximising multi-action tree of depth ``<= 2``."""
    n, p = X.shape
    m = max(1, int(min_leaf_size))
    rows_all = np.arange(n, dtype=np.intp)
    if max_depth <= 0 or n < 2 * m or p == 0:
        return _leaf(R)
    index = _RankIndex(X, split_step=split_step)
    S_tot, C_tot = _accumulate(index, R)
    SA_tot = np.cumsum(S_tot, axis=1)
    SC_tot = np.cumsum(C_tot, axis=1)
    tot = R.sum(axis=0)
    tol = _tie_tol(R)
    best_value, root = _depth1(SA_tot, SC_tot, tot, n, index.cand, m, tol)
    best_tree = _node(X, R, index, rows_all, root)
    if max_depth == 1:
        return best_tree
    A = R.shape[1]
    best_root = None
    for j in range(p):
        order = np.argsort(index.ranks[:, j], kind="stable")
        rank_j = index.ranks[order, j]
        starts = np.flatnonzero(np.r_[True, rank_j[1:] != rank_j[:-1]])
        ends = np.r_[starts[1:], n]
        S = np.zeros((index.p, index.U, A))
        C = np.zeros((index.p, index.U), dtype=np.int64)
        tot_left = np.zeros(A)
        n_left = 0
        n_splits = len(index.uvals[j]) - 1
        for g, r in enumerate(rank_j[starts]):
            if r > n_splits - 1:
                break
            grp = order[starts[g] : ends[g]]
            gr = index.ranks[grp]
            for k in range(index.p):
                np.add.at(S[k], gr[:, k], R[grp])
                np.add.at(C[k], gr[:, k], 1)
            tot_left += R[grp].sum(axis=0)
            n_left += grp.size
            if not index.cand[j, r] or n_left < m or n - n_left < m:
                continue
            SA = np.cumsum(S, axis=1)
            SC = np.cumsum(C, axis=1)
            lv, ls = _depth1(SA, SC, tot_left, n_left, index.cand, m, tol)
            rv, rs = _depth1(
                SA_tot - SA, SC_tot - SC, tot - tot_left, n - n_left, index.cand, m, tol
            )
            if lv == _NEG or rv == _NEG:
                continue
            if lv + rv > best_value + tol:
                best_value = lv + rv
                best_root = (j, int(r), ls, rs)
    if best_root is None:
        return best_tree
    j, r, ls, rs = best_root
    thr = float(index.uvals[j][r])
    left = rows_all[X[:, j] <= thr]
    right = rows_all[X[:, j] > thr]
    return {
        "type": "split",
        "feature": j,
        "threshold": thr,
        "left": _node(X, R, index, left, ls),
        "right": _node(X, R, index, right, rs),
        "n": int(n),
    }


def greedy_multi_action_tree(
    X: np.ndarray,
    R: np.ndarray,
    *,
    max_depth: int,
    min_leaf_size: int,
    split_step: int = 1,
) -> TreeNode:
    """One-step-lookahead tree for depth ``>= 3`` (not the optimum)."""
    n = R.shape[0]
    m = max(1, int(min_leaf_size))
    if max_depth <= 0 or n < 2 * m:
        return _leaf(R)
    best, best_split = -np.inf, None
    for j in range(X.shape[1]):
        vals = np.unique(X[:, j])
        for thr in vals[: len(vals) - 1 : split_step]:
            left = X[:, j] <= thr
            nl = int(left.sum())
            if nl < m or n - nl < m:
                continue
            v = R[left].sum(axis=0).max() + R[~left].sum(axis=0).max()
            if v > best:
                best, best_split = v, (j, float(thr), left)
    if best_split is None or best <= R.sum(axis=0).max():
        return _leaf(R)
    j, thr, left = best_split
    return {
        "type": "split",
        "feature": j,
        "threshold": thr,
        "left": greedy_multi_action_tree(
            X[left],
            R[left],
            max_depth=max_depth - 1,
            min_leaf_size=m,
            split_step=split_step,
        ),
        "right": greedy_multi_action_tree(
            X[~left],
            R[~left],
            max_depth=max_depth - 1,
            min_leaf_size=m,
            split_step=split_step,
        ),
        "n": int(n),
    }


def predict_actions(tree: TreeNode, X: np.ndarray) -> np.ndarray:
    out = np.empty(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        node = tree
        while node["type"] == "split":
            node = (
                node["left"]
                if X[i, node["feature"]] <= node["threshold"]
                else node["right"]
            )
        out[i] = node["action"]
    return out


def tree_rules(
    tree: TreeNode, names: Sequence[str], labels: Sequence[Any], prefix: str = ""
) -> str:
    if tree["type"] == "leaf":
        return (
            f"{prefix}ASSIGN {labels[tree['action']]!r} "
            f"(n={tree['n']}, mean reward={tree['value']:.4f})\n"
        )
    f, t = names[tree["feature"]], tree["threshold"]
    return (
        f"{prefix}IF {f} <= {t:.4f}:\n"
        + tree_rules(tree["left"], names, labels, prefix + "  ")
        + f"{prefix}ELSE ({f} > {t:.4f}):\n"
        + tree_rules(tree["right"], names, labels, prefix + "  ")
    )


def multi_arm_dr_scores(
    Y: np.ndarray,
    G: np.ndarray,
    W: np.ndarray,
    K: int,
    n_folds: int,
    random_state: int,
    clip: float = 0.025,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Cross-fitted AIPW reward matrix ``Gamma[i, a] = mu_a(X_i) +
    1[A_i = a] (Y_i - mu_a(X_i)) / e_a(X_i)``, gradient-boosting
    nuisances (as the binary path), generalized propensity scores clipped
    below at ``clip``. Returns the matrix and the clip count."""
    from sklearn import ensemble
    from sklearn.model_selection import StratifiedKFold

    n = len(Y)
    mu = np.zeros((n, K))
    e = np.zeros((n, K))
    if np.bincount(G, minlength=K).min() < n_folds:
        raise DataInsufficient(
            "policy_tree: a treatment arm has fewer rows than n_folds.",
            recovery_hint="Reduce n_folds, merge sparse arms, or pass scores=.",
        )
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for tr, te in skf.split(W, G):
        for a in range(K):
            sel = tr[G[tr] == a]
            m = ensemble.GradientBoostingRegressor(
                n_estimators=100, max_depth=3, random_state=random_state
            )
            m.fit(W[sel], Y[sel])
            mu[te, a] = m.predict(W[te])
        prop = ensemble.GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=random_state
        )
        prop.fit(W[tr], G[tr])
        e[te] = prop.predict_proba(W[te])
    n_clipped = int((e < clip).sum())
    e = np.maximum(e, clip)
    onehot = G[:, None] == np.arange(K)[None, :]
    Gamma = mu + onehot * (Y[:, None] - mu) / e
    return Gamma, {"n_propensity_clipped": n_clipped, "propensity_clip": clip}


def check_reward_matrix(
    scores: np.ndarray, n: int, n_levels: Optional[int]
) -> np.ndarray:
    R = np.asarray(scores, dtype=float)
    if R.ndim != 2 or R.shape[0] != n:
        raise MethodIncompatibility(
            f"policy_tree: a multi-action reward matrix must be (n, A) with "
            f"n = {n}; got shape {R.shape}.",
            recovery_hint="Pass one row per input row and one column per action.",
        )
    if n_levels is not None and R.shape[1] != n_levels:
        raise MethodIncompatibility(
            f"policy_tree: the reward matrix has {R.shape[1]} columns but the "
            f"treatment has {n_levels} levels.",
            recovery_hint="Order the columns as the sorted treatment levels.",
        )
    if R.shape[1] < 2:
        raise MethodIncompatibility("policy_tree: need at least two actions.")
    if not np.isfinite(R).all():
        raise MethodIncompatibility(
            "policy_tree: the reward matrix has non-finite values."
        )
    return R


def value_summary(R: np.ndarray, actions: np.ndarray, alpha: float) -> Dict[str, Any]:
    from scipy import stats

    n = R.shape[0]
    contrib = R[np.arange(n), actions]
    v = float(contrib.mean())
    se = float(contrib.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    z = float(stats.norm.ppf(1 - alpha / 2))
    means = R.mean(axis=0)
    best = int(np.argmax(means))
    gain_contrib = contrib - R[:, best]
    return {
        "value_policy": v,
        "value_policy_se": se,
        "value_policy_ci": (v - z * se, v + z * se),
        "action_values": means,
        "best_uniform_action": best,
        "value_gain": float(gain_contrib.mean()),
        "value_gain_se": (
            float(gain_contrib.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        ),
    }


def fit_multi_action(
    est: Any,
    X_pol: np.ndarray,
    R: np.ndarray,
    labels: List[Any],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """Grow, predict and summarise; returns the fields of the result."""
    tree, mode = grow_multi_action_tree(est, X_pol, R)
    actions = predict_actions(tree, X_pol)
    out = value_summary(R, actions, est.alpha)
    out.update(
        tree=tree,
        search_mode=mode,
        actions=actions,
        policy=np.asarray(labels, dtype=object)[actions],
        action_shares={
            str(lab): float(np.mean(actions == a)) for a, lab in enumerate(labels)
        },
        rules=tree_rules(tree, est.policy_covariates, labels),
    )
    out.update(extra)
    return out


def grow_multi_action_tree(
    est: Any, X_pol: np.ndarray, R: np.ndarray
) -> Tuple[TreeNode, str]:
    """The multi-action tree for ``est``'s settings and its search mode."""
    n, p = X_pol.shape
    if est.max_depth <= 2 and est.search != "greedy":
        from ._exact_tree import exact_search_is_affordable, warn_greedy_fallback

        if est.search == "auto" and not exact_search_is_affordable(
            n, p * R.shape[1], est.max_depth
        ):
            warn_greedy_fallback(n, p, est.max_depth)
            tree, mode = (
                greedy_multi_action_tree(
                    X_pol,
                    R,
                    max_depth=est.max_depth,
                    min_leaf_size=est.min_leaf_size,
                    split_step=est.split_step,
                ),
                "greedy",
            )
        else:
            tree, mode = (
                exact_multi_action_tree(
                    X_pol,
                    R,
                    max_depth=est.max_depth,
                    min_leaf_size=est.min_leaf_size,
                    split_step=est.split_step,
                ),
                "exact",
            )
    else:
        tree, mode = (
            greedy_multi_action_tree(
                X_pol,
                R,
                max_depth=est.max_depth,
                min_leaf_size=est.min_leaf_size,
                split_step=est.split_step,
            ),
            "greedy",
        )
    return tree, mode


def honest_value_summary(
    grow: Any,
    predict: Any,
    X: np.ndarray,
    R: np.ndarray,
    n_folds: int,
    random_state: int,
    alpha: float,
) -> Dict[str, Any]:
    """Cross-fitted value of the policy-learning procedure.

    The rows are split into ``n_folds`` folds; on each, the tree is grown on
    the other folds (``grow(X, R) -> tree``) and its actions on the held-out
    rows are evaluated with their own rewards, and so is the best uniform
    action of the training folds. A policy chosen to maximise the in-sample
    mean of ``R[i, pi(X_i)]`` and evaluated on the same rows overstates its
    value (under rewards that do not depend on the action, the in-sample
    gain over the best uniform action is positive by construction); every
    unit here is evaluated by a policy that did not see it, so the
    influence-function standard error describes the estimate.
    """
    import warnings

    from scipy import stats
    from sklearn.model_selection import KFold

    n = R.shape[0]
    actions = np.empty(n, dtype=int)
    uniform = np.empty(n, dtype=int)
    for tr, te in KFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    ).split(X):
        with warnings.catch_warnings():
            # The full-sample fit has already reported any search fallback.
            warnings.simplefilter("ignore")
            tree = grow(X[tr], R[tr])
        actions[te] = predict(tree, X[te])
        uniform[te] = int(np.argmax(R[tr].mean(axis=0)))
    rows = np.arange(n)
    contrib = R[rows, actions]
    gain_contrib = contrib - R[rows, uniform]
    v = float(contrib.mean())
    se = float(contrib.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    z = float(stats.norm.ppf(1 - alpha / 2))
    return {
        "value_policy": v,
        "value_policy_se": se,
        "value_policy_ci": (v - z * se, v + z * se),
        "value_gain": float(gain_contrib.mean()),
        "value_gain_se": (
            float(gain_contrib.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        ),
        "oof_actions": actions,
    }
