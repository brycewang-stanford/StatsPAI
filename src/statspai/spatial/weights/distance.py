"""Distance-based spatial weights: KNN, distance band, kernel."""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial import cKDTree

from .core import W


def knn_weights(coords: np.ndarray, k: int = 5) -> W:
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if k <= 0 or k >= n:
        raise ValueError(f"k must be in (0, n); got k={k}, n={n}")
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    neighbors = {i: [int(j) for j in idx[i, 1:]] for i in range(n)}
    return W(neighbors)


def distance_band(coords: np.ndarray, threshold: float, binary: bool = True) -> W:
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
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
            k_vals = np.exp(-0.5 * u ** 2)
        elif kernel == "bisquare":
            k_vals = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
        elif kernel == "triangular":
            k_vals = np.where(u < 1, 1 - u, 0.0)
        else:
            raise ValueError(f"unknown kernel {kernel!r}")
        neighbors[i] = js
        weights[i] = k_vals.tolist()
    return W(neighbors, weights)
