"""Nearest neighbours on a covariate metric without the full distance matrix.

Matching on the Mahalanobis or the standardised Euclidean distance used to
compute every target-to-pool distance (``scipy.spatial.distance.cdist``,
in row blocks) and read each target's nearest units off its row; with
29,000 treated and 71,000 control units that is two billion distances for
the matches and six billion more for the same-arm neighbours of the
Abadie-Imbens variance, almost three minutes in all.

Here a k-d tree on whitened coordinates (``z = x L`` with ``L L' = VI``,
so that Euclidean distance between ``z`` equals the Mahalanobis distance
between ``x``) finds, for each target, a small set of units that contains
every unit the full row could select. Only those distances are then
computed, with the same ``cdist`` call on the same inputs; ``cdist``
evaluates each pair on its own, so they carry the bits the full row would
hold, and the selection rules are applied to them unchanged.

The tree's distances differ from ``cdist``'s by rounding. ``slack`` bounds
that difference with a wide margin (below), and every radius the tree is
asked for is widened by it, so a candidate set can only be too large,
which costs time, never too small, which would change a match.

Rounding: ``cdist`` forms ``d^2 = u' VI u`` for ``u = x_i - x_j``, which
rounds by at most ``(p + 2) eps |u|' |VI| |u|`` and hence moves ``d`` by
at most ``sqrt((p + 2) eps sqrt(p) ||VI||) |u|`` (the square root is the
worst case, reached when ``d`` itself is near zero); ``L L'`` differs from
``VI`` by the Cholesky backward error ``O(p eps ||VI||)``, which moves the
tree's distance by an amount of the same form, and the whitened
coordinates round by ``O(p eps ||L|| |x|)``. With ``||L|| = sqrt(||VI||)``
and ``|u| <= 2 sqrt(p) max|x|`` each term is below
``3e-7 * 2 sqrt(p) ||L|| max|x|`` for ``p <= 16``; ``slack`` is
``1e-5 * 2 sqrt(p) ||L|| max|x|``, more than ten times their sum.

The fast path is taken only where it is known to be exact: finite
covariates of moderate size, a metric whose matrix is positive definite
with condition number below ``1e8`` (so no squared distance can round to a
negative number and turn into ``nan``, which sends the block-wise search
down its row-by-row branch), at most ``_MAX_DIM`` covariates (beyond that
a k-d tree prunes little), and a problem large enough to be worth it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

#: Most covariates for which the tree is used.
_MAX_DIM = 16

#: Smallest ``n_target * n_pool`` for which the tree is used; below it the
#: full rows are cheap. Tests lower it to run the tree on small designs.
_MIN_CELLS = 1 << 22

#: Largest condition number of the metric matrix for which the tree is used.
_MAX_COND = 1e8

#: Largest covariate magnitude for which the tree is used.
_MAX_ABS = 1e100

#: Targets handled per batch of tree queries.
_BATCH = 4096


class MetricPoints:
    """Two groups of points under one ``cdist`` metric, with a k-d tree.

    ``A`` and ``B`` are the arrays ``cdist`` is called with (targets and
    pool), ``kw`` its keyword arguments; ``za`` and ``zb`` are the whitened
    coordinates and ``slack`` the bound on the difference between the two
    distances.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        metric: str,
        kw: Dict[str, Any],
        za: np.ndarray,
        zb: np.ndarray,
        slack: float,
    ) -> None:
        self.A = A
        self.B = B
        self.metric = metric
        self.kw = kw
        self.za = za
        self.zb = zb
        self.slack = float(slack)
        self._tree: Optional[cKDTree] = None

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.zb)
        return self._tree

    def exact(self, i: int, cols: np.ndarray) -> np.ndarray:
        """``cdist`` distances of target ``i`` to pool units ``cols``."""
        return np.asarray(
            cdist(self.A[i : i + 1], self.B[cols], metric=self.metric, **self.kw)[0],
            dtype=float,
        )


def metric_points(
    X_from: np.ndarray,
    X_to: np.ndarray,
    distance: str,
    *,
    VI: Optional[np.ndarray] = None,
    sd: Optional[np.ndarray] = None,
    force: bool = False,
) -> Optional[MetricPoints]:
    """The tree for a target and a pool group, or ``None`` where it is not exact.

    ``distance='mahalanobis'`` takes the matrix ``VI`` ``cdist`` is given;
    ``'euclidean'`` takes the standard deviations the covariates are
    divided by. ``force`` skips the size threshold only.
    """
    X_from = np.asarray(X_from, dtype=float)
    X_to = np.asarray(X_to, dtype=float)
    if X_from.ndim != 2 or X_to.ndim != 2 or X_from.shape[1] != X_to.shape[1]:
        return None
    n_from, p = X_from.shape
    n_to = X_to.shape[0]
    if n_from == 0 or n_to == 0 or p == 0 or p > _MAX_DIM:
        return None
    if not force and n_from * n_to < _MIN_CELLS:
        return None
    if not (np.all(np.isfinite(X_from)) and np.all(np.isfinite(X_to))):
        return None
    if distance == "mahalanobis":
        if VI is None:
            return None
        VI = np.asarray(VI, dtype=float)
        if VI.shape != (p, p) or not np.all(np.isfinite(VI)):
            return None
        S = (VI + VI.T) / 2.0
        eig = np.linalg.eigvalsh(S)
        if not (eig[0] > 0 and eig[0] > eig[-1] / _MAX_COND):
            return None
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            return None
        A, B = X_from, X_to
        norm_L = float(np.sqrt(eig[-1]))
        centre = np.mean(np.vstack([A.mean(axis=0), B.mean(axis=0)]), axis=0)
        za = (A - centre) @ L
        zb = (B - centre) @ L
        kw: Dict[str, Any] = {"VI": VI}
    elif distance == "euclidean":
        if sd is None:
            return None
        sd = np.asarray(sd, dtype=float)
        A = X_from / sd
        B = X_to / sd
        norm_L = 1.0
        za, zb = A, B
        kw = {}
    else:
        return None
    big = max(float(np.max(np.abs(A))), float(np.max(np.abs(B))))
    if not np.isfinite(big) or big > _MAX_ABS:
        return None
    if not (np.all(np.isfinite(za)) and np.all(np.isfinite(zb))):
        return None
    slack = 1e-5 * 2.0 * np.sqrt(p) * norm_L * big + 1e-300
    return MetricPoints(A, B, distance, kw, za, zb, slack)


def nearest_candidates(
    points: MetricPoints,
    n_matches: int,
    kth_distance: Callable[..., Tuple[np.ndarray, np.ndarray]],
    candidate_radius: Callable[[np.ndarray], np.ndarray],
    caliper: Optional[float],
) -> Iterator[Tuple[Any, ...]]:
    """What ``_block_candidates`` yields, from the tree.

    For each target: the ``kk`` nearest pool distances (``kth_distance``
    turns them into the number of matches ``k`` and the distance ``kth`` of
    the last), and every pool unit within ``candidate_radius(kth)``, in
    ascending column order with its ``cdist`` distance.
    """
    n_target = points.A.shape[0]
    n_pool = points.B.shape[0]
    kk = min(n_matches, n_pool)
    e = points.slack
    tree = points.tree
    for start in range(0, n_target, _BATCH):
        stop = min(start + _BATCH, n_target)
        n_rows = stop - start
        if kk == 0:
            yield n_rows, None, None, None, None, None, points_block(
                points, start, stop
            )
            continue
        dq, _ = tree.query(points.za[start:stop], k=kk)
        r_tree = np.asarray(dq, dtype=float).reshape(n_rows, kk)[:, -1]
        # the exact k-th distance is at most r_tree + e, so the radius the
        # selection will use is at most candidate_radius(r_tree + e)
        reach = candidate_radius(r_tree + e) + e
        balls = tree.query_ball_point(points.za[start:stop], reach)
        # A caliper narrower than the k-th distance can leave fewer than kk
        # candidates; the missing places are farther than the caliper, so
        # they count as no match, as the farther units of the full row do.
        nearest = np.full((n_rows, kk), np.inf)
        cand_cols = []
        cand_d = []
        for r in range(n_rows):
            cols = np.asarray(balls[r], dtype=np.intp)
            cols.sort()
            d = points.exact(start + r, cols)
            m = min(kk, d.size)
            if m == 1:
                nearest[r, 0] = d.min()
            elif m > 1:
                nearest[r, :m] = np.sort(np.partition(d, m - 1)[:m])
            cand_cols.append(cols)
            cand_d.append(d)
        k, kth = kth_distance(nearest, caliper)
        radius = candidate_radius(kth)
        radius[k == 0] = -1.0
        rows_l = []
        cols_l = []
        dv_l = []
        for r in range(n_rows):
            keep = cand_d[r] <= radius[r]
            c = cand_cols[r][keep]
            rows_l.append(np.full(c.size, r, dtype=np.intp))
            cols_l.append(c)
            dv_l.append(cand_d[r][keep])
        rows = np.concatenate(rows_l) if rows_l else np.zeros(0, dtype=np.intp)
        cols = np.concatenate(cols_l) if cols_l else np.zeros(0, dtype=np.intp)
        dv = np.concatenate(dv_l) if dv_l else np.zeros(0)
        yield n_rows, rows, cols, dv, kth, k, None


def points_block(points: MetricPoints, start: int, stop: int) -> np.ndarray:
    """Full ``cdist`` rows ``start:stop`` (the row-by-row fallback's input)."""
    return np.asarray(
        cdist(points.A[start:stop], points.B, metric=points.metric, **points.kw),
        dtype=float,
    )


def same_arm_neighbours(
    points: MetricPoints, j: int, ties_all: bool
) -> Iterator[Tuple[int, np.ndarray]]:
    """Each unit's ``j`` nearest other units of one group, with ties.

    ``points`` holds the group as both targets and pool. Yields, for every
    unit with a neighbour, its position and the positions of its
    neighbours in the order the full row would give them: under
    ``ties_all`` every unit at most as far as the ``j``-th nearest, in
    ascending position; otherwise the ``j`` nearest by distance, ties to
    the lower position.
    """
    n = points.A.shape[0]
    k = min(j, n - 1)
    if k <= 0:
        return
    e = points.slack
    tree = points.tree
    for start in range(0, n, _BATCH):
        stop = min(start + _BATCH, n)
        # the k + 1 nearest hold at least k units other than the unit
        # itself, so (on the tree's distances) its k-th nearest other unit
        # is no farther than the (k + 1)-th returned
        dq, _ = tree.query(points.za[start:stop], k=k + 1)
        r_tree = np.asarray(dq, dtype=float).reshape(stop - start, k + 1)[:, -1]
        balls = tree.query_ball_point(points.za[start:stop], r_tree + 2.0 * e)
        for r in range(stop - start):
            row = start + r
            cols = np.asarray(balls[r], dtype=np.intp)
            cols = cols[cols != row]
            cols.sort()
            d = points.exact(row, cols)
            kth = np.partition(d, k - 1)[k - 1]
            cand = np.flatnonzero(d <= kth)
            if not ties_all:
                cand = cand[np.argsort(d[cand], kind="stable")][:k]
            yield row, cols[cand]
