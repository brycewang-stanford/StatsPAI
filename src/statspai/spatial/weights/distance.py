"""Distance-based spatial weights: KNN, distance band, kernel."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial import cKDTree

from .core import W


def knn_weights(coords: np.ndarray, k: int = 5) -> W:
    """k-nearest-neighbour spatial weights from point coordinates.

    Each unit is connected to its ``k`` closest units by Euclidean distance
    (the unit itself is excluded). The resulting graph is generally asymmetric
    — being someone's nearest neighbour is not reciprocal.

    Parameters
    ----------
    coords : numpy.ndarray
        ``(n, d)`` array of coordinates (typically ``d=2``).
    k : int, default 5
        Number of nearest neighbours per unit. Must satisfy ``0 < k < n``.

    Returns
    -------
    W
        Spatial weights object with exactly ``k`` neighbours per unit.

    Raises
    ------
    ValueError
        If ``k`` is not in the open interval ``(0, n)``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> coords = np.random.default_rng(0).random((30, 2))
    >>> w = sp.knn_weights(coords, k=4)
    >>> len(w.neighbors[0])
    4

    See Also
    --------
    distance_band : Connect all units within a fixed radius.
    kernel_weights : Continuous distance-decay weights.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if k <= 0 or k >= n:
        raise ValueError(f"k must be in (0, n); got k={k}, n={n}")
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    neighbors = {i: [int(j) for j in idx[i, 1:]] for i in range(n)}
    return W(neighbors)


def distance_band(coords: np.ndarray, threshold: float, binary: bool = True) -> W:
    """Distance-band spatial weights — connect units within a fixed radius.

    Every unit is connected to all other units within Euclidean distance
    ``threshold``. Weights are either binary (1 for every neighbour) or
    inverse-distance (``1/d``).

    Parameters
    ----------
    coords : numpy.ndarray
        ``(n, d)`` array of coordinates.
    threshold : float
        Distance band radius. Units within this distance become neighbours.
        Choose at least the maximum nearest-neighbour distance to avoid
        islands (units with no neighbours).
    binary : bool, default True
        If True all neighbour weights are ``1.0``; if False they are inverse
        Euclidean distance ``1/d`` (coincident points are dropped via an
        infinite distance).

    Returns
    -------
    W
        Spatial weights object; neighbour counts vary by local density.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> coords = np.random.default_rng(0).random((30, 2))
    >>> w = sp.distance_band(coords, threshold=0.3, binary=False)
    >>> w.n
    30

    See Also
    --------
    knn_weights : Fixed neighbour count instead of fixed radius.
    """
    coords = np.asarray(coords, dtype=float)
    tree = cKDTree(coords)
    pairs = tree.query_ball_point(coords, r=threshold)
    neighbors, weights = {}, {}
    for i, js in enumerate(pairs):
        js = [int(j) for j in js if j != i]
        neighbors[i] = js
        if binary:
            weights[i] = [1.0] * len(js)
        else:
            if js:
                d = np.linalg.norm(coords[js] - coords[i], axis=1)
                d[d == 0] = np.inf
                weights[i] = (1.0 / d).tolist()
            else:
                weights[i] = []
    return W(neighbors, weights)


def kernel_weights(
    coords: np.ndarray,
    bandwidth: float,
    kernel: Literal["gaussian", "bisquare", "triangular"] = "gaussian",
    fixed: bool = True,
) -> W:
    """Kernel (distance-decay) spatial weights from point coordinates.

    Assigns each neighbour a continuous weight that decays with distance
    according to a kernel function. Supports a fixed bandwidth (same radius
    everywhere) or an adaptive bandwidth (the ``bandwidth``-th nearest
    neighbour distance, varying with local density).

    Parameters
    ----------
    coords : numpy.ndarray
        ``(n, d)`` array of coordinates.
    bandwidth : float
        With ``fixed=True``, the bandwidth distance ``h`` (for the bisquare
        and triangular kernels, neighbours beyond ``h`` get zero weight; the
        gaussian kernel has infinite support so all units are included).
        With ``fixed=False``, the integer number of nearest neighbours that
        defines the adaptive bandwidth.
    kernel : {"gaussian", "bisquare", "triangular"}, default "gaussian"
        Kernel shape. ``gaussian`` ``exp(-u^2/2)``; ``bisquare``
        ``(1-u^2)^2`` for ``u<1``; ``triangular`` ``1-u`` for ``u<1``, where
        ``u = d / bandwidth``.
    fixed : bool, default True
        Fixed (True) versus adaptive (False) bandwidth.

    Returns
    -------
    W
        Spatial weights object with continuous kernel weights.

    Raises
    ------
    ValueError
        If ``kernel`` is not one of the supported names.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> coords = np.random.default_rng(0).random((30, 2))
    >>> w = sp.kernel_weights(coords, bandwidth=0.4, kernel="bisquare")
    >>> w.n
    30

    See Also
    --------
    distance_band : Binary / inverse-distance fixed-radius weights.
    knn_weights : Uniform k-nearest-neighbour weights.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    tree = cKDTree(coords)
    neighbors, weights = {}, {}
    for i in range(n):
        if fixed:
            if kernel == "gaussian":
                # Gaussian has infinite support; include all other points.
                js = [int(j) for j in range(n) if j != i]
            else:
                js = tree.query_ball_point(coords[i], r=bandwidth)
                js = [int(j) for j in js if j != i]
            if js:
                d = np.linalg.norm(coords[js] - coords[i], axis=1)
            else:
                d = np.array([])
            bw = bandwidth
        else:
            dists, idx = tree.query(coords[i], k=int(bandwidth) + 1)
            js = [int(j) for j in idx[1:]]
            d = dists[1:]
            bw = d.max() if len(d) else 1.0
        u = d / bw if bw > 0 else np.zeros_like(d)
        if kernel == "gaussian":
            k_vals = np.exp(-0.5 * u**2)
        elif kernel == "bisquare":
            k_vals = np.where(u < 1, (1 - u**2) ** 2, 0.0)
        elif kernel == "triangular":
            k_vals = np.where(u < 1, 1 - u, 0.0)
        else:
            raise ValueError(f"unknown kernel {kernel!r}")
        neighbors[i] = js
        weights[i] = k_vals.tolist()
    return W(neighbors, weights)
