"""Correctness test for the vectorized Conley (1999) spatial-HAC sandwich.

The meat accumulation in ``sp.conley`` was rewritten from per-observation /
per-pair Python ``np.outer`` loops to vectorized matrix products
(``Xe.T @ Xe`` for the diagonal, ``M + M.T`` for the pairs). This test pins
the vectorized output to an *independent* explicit-loop reference computation
of the same estimator, so any algebraic drift is caught.
"""

import numpy as np
import pandas as pd

import statspai as sp
from statspai.inference.conley import (
    _haversine_km,
    _latlon_to_cartesian,
)
from scipy.spatial import cKDTree

_EARTH_RADIUS_KM = 6371.0088


def _reference_conley_se(result, data, lat, lon, dist_cutoff, kernel):
    """Naive O(n^2)-ish reference using explicit per-pair outer products."""
    X = np.asarray(result.data_info["X"])
    resid = np.asarray(result.data_info["residuals"])
    n, k = X.shape
    lat_v = data[lat].values.astype(float)
    lon_v = data[lon].values.astype(float)
    XtX_inv = np.linalg.inv(X.T @ X)
    Xe = X * resid[:, None]

    coords = _latlon_to_cartesian(lat_v, lon_v)
    tree = cKDTree(coords)
    theta = dist_cutoff / _EARTH_RADIUS_KM
    chord = 2 * _EARTH_RADIUS_KM * np.sin(theta / 2)

    Omega = np.zeros((k, k))
    for i in range(n):
        Omega += np.outer(Xe[i], Xe[i])
    pairs = tree.query_pairs(r=chord, output_type="ndarray")
    for p in range(len(pairs)):
        i, j = pairs[p]
        d = _haversine_km(lat_v[i], lon_v[i], lat_v[j], lon_v[j])
        if d > dist_cutoff:
            continue
        w = 1.0 if kernel == "uniform" else (1.0 - d / dist_cutoff)
        cross = np.outer(Xe[i], Xe[j]) + np.outer(Xe[j], Xe[i])
        Omega += w * cross
    V = XtX_inv @ Omega @ XtX_inv
    return np.sqrt(np.diag(V))


def _fixture(n=500, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "latitude": rng.uniform(30, 45, n),
            "longitude": rng.uniform(-120, -80, n),
        }
    )
    df["y"] = 1 + 0.5 * df.x1 - 0.3 * df.x2 + rng.normal(size=n)
    return df


def test_conley_uniform_matches_reference_loop():
    df = _fixture()
    res = sp.regress("y ~ x1 + x2", data=df)
    c = sp.conley(
        res, data=df, lat="latitude", lon="longitude", dist_cutoff=400, kernel="uniform"
    )
    ref = _reference_conley_se(res, df, "latitude", "longitude", 400, "uniform")
    np.testing.assert_allclose(c.std_errors.values, ref, rtol=1e-10, atol=1e-12)


def test_conley_bartlett_matches_reference_loop():
    df = _fixture(seed=7)
    res = sp.regress("y ~ x1 + x2", data=df)
    c = sp.conley(
        res,
        data=df,
        lat="latitude",
        lon="longitude",
        dist_cutoff=350,
        kernel="bartlett",
    )
    ref = _reference_conley_se(res, df, "latitude", "longitude", 350, "bartlett")
    np.testing.assert_allclose(c.std_errors.values, ref, rtol=1e-10, atol=1e-12)
