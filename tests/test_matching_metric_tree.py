"""The k-d tree search gives the matches the full distance rows give, bit for bit.

Matching on the Mahalanobis or the standardised Euclidean distance reads
each target's nearest units off a k-d tree instead of computing every
target-to-pool distance (``matching/_metric_tree.py``). The candidates'
distances come from the same ``cdist`` call, so every number the fit
reports must be the one the full rows gave, ties included. The full-row
search is still in the code (it runs below the size threshold and wherever
the tree is not exact), which is what the fits are compared against; the
same-arm neighbours of the Abadie-Imbens variance are compared against the
loop ``_same_arm_outcome`` ran through 1.39.6.
"""

import hashlib

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

import statspai as sp
from statspai.matching import _metric_tree


def _fingerprint(result) -> str:
    """Bit-exact digest of the estimate, the inference and the detail table."""
    h = hashlib.sha256()
    for v in (result.estimate, result.se, *result.ci, result.pvalue):
        h.update(np.float64(v).tobytes())
    detail = getattr(result, "detail", None)
    if isinstance(detail, pd.DataFrame):
        for c in detail.columns:
            h.update(str(c).encode())
            col = detail[c].to_numpy()
            if col.dtype.kind in "fiub":
                h.update(np.ascontiguousarray(col, dtype=float).tobytes())
            else:
                h.update(repr(col.tolist()).encode())
    return h.hexdigest()


def _design(rep: int):
    rng = np.random.default_rng(1000 + rep)
    n = int(rng.integers(40, 400))
    p = int(rng.integers(1, 6))
    X = rng.normal(size=(n, p)) * rng.uniform(0.2, 50, p) + rng.normal(size=p) * 10
    kind = rep % 5
    if kind == 1:  # many exact ties
        X = rng.integers(0, 3, size=(n, p)).astype(float)
    elif kind == 2:  # duplicated rows
        X[n // 2 :] = X[: n - n // 2]
    elif kind == 3:  # integers far from zero
        X = np.round(X) + 1e4
    elif kind == 4:  # a binary column
        X[:, 0] = rng.integers(0, 2, n)
    z = (X[:, 0] - X[:, 0].mean()) / (X[:, 0].std() + 1e-9)
    d = (rng.uniform(size=n) < np.clip(1 / (1 + np.exp(-z)), 0.15, 0.85)).astype(int)
    d[:3] = 1
    d[-3:] = 0
    y = X.sum(axis=1) * 0.1 + 2 * d + rng.normal(size=n)
    df = pd.DataFrame(X, columns=[f"x{k}" for k in range(p)])
    df["y"] = np.round(y) if rep % 4 == 0 else y
    df["d"] = d
    opts = dict(
        distance=["mahalanobis", "euclidean"][rep % 2],
        estimand=["ATT", "ATE"][(rep // 2) % 2],
        n_matches=[1, 2, 4][(rep // 4) % 3],
        ties=["all", "first"][(rep // 3) % 2],
    )
    if rep % 5 == 0:
        opts["bias_correction"] = True
    if rep % 7 == 0:
        opts["caliper"] = float(rng.uniform(0.2, 2.0))
    if rep % 11 == 0:
        opts["tie_tolerance"] = 1e-3
    if rep % 13 == 0:
        opts["ai_matches"] = 3
    if rep % 9 == 0:
        opts["se_method"] = "abadie_imbens_pop"
    return df, [f"x{k}" for k in range(p)], opts


def _fit(df, covs, opts):
    try:
        return _fingerprint(sp.match(df, y="y", treat="d", covariates=covs, **opts))
    except sp.exceptions.StatsPAIError as exc:
        return f"{type(exc).__name__}: {exc}"


@pytest.mark.parametrize("rep", range(48))
def test_tree_search_reproduces_the_full_rows(rep, monkeypatch):
    df, covs, opts = _design(rep)
    calls = {"tree": 0}
    real = _metric_tree.metric_points

    def counted(*args, **kwargs):
        out = real(*args, **kwargs)
        calls["tree"] += out is not None
        return out

    monkeypatch.setattr(_metric_tree, "metric_points", lambda *a, **k: None)
    full_rows = _fit(df, covs, opts)

    monkeypatch.setattr(_metric_tree, "metric_points", counted)
    monkeypatch.setattr(_metric_tree, "_MIN_CELLS", 0)
    monkeypatch.setattr(_metric_tree, "_BATCH", 37)  # cross batch boundaries
    tree = _fit(df, covs, opts)

    assert tree == full_rows
    assert calls["tree"] > 0


def _same_arm_loop(X, VI, j, ties_all):
    """The neighbour search of ``_same_arm_outcome`` through 1.39.6."""
    out = {}
    d_all = cdist(X, X, metric="mahalanobis", VI=VI)
    for row in range(len(X)):
        d = d_all[row].copy()
        d[row] = np.inf
        k = min(j, int(np.sum(np.isfinite(d))))
        if k == 0:
            continue
        kth = np.partition(d, k - 1)[k - 1]
        cand = np.flatnonzero(d <= kth)
        if not ties_all:
            cand = cand[np.argsort(d[cand], kind="stable")][:k]
        out[row] = cand
    return out


@pytest.mark.parametrize("ties_all", [True, False])
@pytest.mark.parametrize("j", [1, 3])
@pytest.mark.parametrize("kind", ["continuous", "discrete", "duplicates"])
def test_same_arm_neighbours_equal_the_loop(kind, j, ties_all):
    rng = np.random.default_rng(7)
    n, p = 300, 3
    if kind == "discrete":
        X = rng.integers(0, 3, size=(n, p)).astype(float)
    else:
        X = rng.normal(size=(n, p)) * [1.0, 30.0, 0.01] + [0.0, 500.0, 2.0]
        if kind == "duplicates":
            X[150:] = X[:150]
    M = rng.normal(size=(p, p))
    VI = np.linalg.inv(M @ M.T + np.eye(p))
    points = _metric_tree.metric_points(X, X, "mahalanobis", VI=VI, force=True)
    assert points is not None
    tree = dict(_metric_tree.same_arm_neighbours(points, j, ties_all))
    loop = _same_arm_loop(X, VI, j, ties_all)
    assert tree.keys() == loop.keys()
    for row, cand in loop.items():
        np.testing.assert_array_equal(tree[row], cand)


def test_exact_distances_are_the_full_row_bits():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(20, 4)) * 100
    B = rng.normal(size=(500, 4)) * 100
    VI = np.linalg.inv(np.cov(np.vstack([A, B]).T))
    points = _metric_tree.metric_points(A, B, "mahalanobis", VI=VI, force=True)
    full = cdist(A, B, metric="mahalanobis", VI=VI)
    for i in range(len(A)):
        cols = np.sort(rng.choice(len(B), size=50, replace=False))
        np.testing.assert_array_equal(points.exact(i, cols), full[i, cols])


@pytest.mark.parametrize(
    "case",
    ["singular", "ill_conditioned", "too_many_columns", "non_finite", "small"],
)
def test_tree_is_not_used_where_it_is_not_exact(case):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    VI = np.eye(3)
    force = True
    if case == "singular":
        VI = np.diag([1.0, 1.0, 0.0])
    elif case == "ill_conditioned":
        VI = np.diag([1.0, 1.0, 1e-9])
    elif case == "too_many_columns":
        X = rng.normal(size=(200, _metric_tree._MAX_DIM + 1))
        VI = np.eye(X.shape[1])
    elif case == "non_finite":
        X[5, 1] = np.inf
    elif case == "small":
        force = False
    assert _metric_tree.metric_points(X, X, "mahalanobis", VI=VI, force=force) is None
