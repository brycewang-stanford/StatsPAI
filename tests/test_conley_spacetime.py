"""Behaviour tests for the ``sp.conley`` spatial + time HAC extension.

Covers, in order:

* **backwards compatibility** — the pre-existing spatial-only call path must
  be bit-identical to the original algorithm;
* **failing loudly** — half-specified space-time requests, non-integer periods
  and units whose coordinates move must raise, never be silently ignored;
* **kernel semantics** — the same-unit / cross-unit split and the bandwidths;
* **memory** — the panel path must stay O(neighbour pairs), never O(n^2).
"""

from __future__ import annotations

import tracemalloc
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp
    from statspai.exceptions import MethodIncompatibility
    from statspai.inference.conley import (
        _EARTH_RADIUS_KM,
        _haversine_km,
        _latlon_to_cartesian,
    )


@pytest.fixture(scope="module")
def cross_section():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "lat": rng.uniform(30, 45, size=n),
            "lon": rng.uniform(-120, -100, size=n),
        }
    )
    df["y"] = 1.0 + 0.5 * df.x1 - 0.3 * df.x2 + rng.normal(size=n)
    return df


@pytest.fixture(scope="module")
def geo_panel():
    rng = np.random.default_rng(11)
    n_units, n_periods = 40, 15
    lat0 = rng.uniform(35.0, 42.0, n_units)
    lon0 = rng.uniform(-110.0, -100.0, n_units)
    rows = [(u, t, lat0[u], lon0[u]) for u in range(n_units) for t in range(n_periods)]
    df = pd.DataFrame(rows, columns=["id", "t", "lat", "lon"])
    n = len(df)
    df["x1"] = rng.normal(size=n)
    df["x2"] = rng.normal(size=n)
    df["y"] = 1.0 + 0.5 * df.x1 - 0.3 * df.x2 + rng.normal(size=n)
    return df


# --------------------------------------------------------------------------
# 1. Backwards compatibility
# --------------------------------------------------------------------------


def _legacy_conley_vcov(result, data, lat, lon, dist_cutoff, kernel):
    """Verbatim reimplementation of the original spatial-only algorithm.

    Kept independent of the production code on purpose: if someone refactors
    ``sp.conley`` and perturbs the default path by even one ULP, this fails.
    """
    X = np.asarray(result.data_info["X"])
    residuals = np.asarray(result.data_info["residuals"])
    lat_vals = data[lat].values.astype(float)
    lon_vals = data[lon].values.astype(float)
    XtX_inv = np.linalg.inv(X.T @ X)

    tree = cKDTree(_latlon_to_cartesian(lat_vals, lon_vals))
    theta = dist_cutoff / _EARTH_RADIUS_KM
    chord_cutoff = 2 * _EARTH_RADIUS_KM * np.sin(theta / 2)

    Xe = X * residuals[:, np.newaxis]
    Omega = Xe.T @ Xe
    pairs = tree.query_pairs(r=chord_cutoff, output_type="ndarray")
    if len(pairs) > 0:
        idx_i, idx_j = pairs[:, 0], pairs[:, 1]
        d_ij = _haversine_km(
            lat_vals[idx_i], lon_vals[idx_i], lat_vals[idx_j], lon_vals[idx_j]
        )
        within = d_ij <= dist_cutoff
        weights = (
            np.ones_like(d_ij) if kernel == "uniform" else 1.0 - d_ij / dist_cutoff
        )
        weights = weights * within
        M = (Xe[idx_i] * weights[:, np.newaxis]).T @ Xe[idx_j]
        Omega += M + M.T
    return XtX_inv @ Omega @ XtX_inv


@pytest.mark.parametrize("kernel", ["uniform", "bartlett"])
@pytest.mark.parametrize("cutoff", [50.0, 200.0, 800.0])
def test_default_path_is_bit_identical_to_the_original(cross_section, kernel, cutoff):
    """Existing calls must be unchanged to the last bit. Non-negotiable."""
    r = sp.regress("y ~ x1 + x2", data=cross_section)
    got = sp.conley(
        r, cross_section, "lat", "lon", dist_cutoff=cutoff, kernel=kernel
    ).data_info["vcov"]
    want = _legacy_conley_vcov(r, cross_section, "lat", "lon", cutoff, kernel)
    assert np.array_equal(got, want), "spatial-only path drifted from the original"


def test_default_signature_still_positional(cross_section):
    """The historical positional call must keep working."""
    r = sp.regress("y ~ x1 + x2", data=cross_section)
    res = sp.conley(r, cross_section, "lat", "lon", 200.0, "bartlett", 0.05)
    assert res.model_info["se_type"] == "conley_spatial"
    assert np.all(np.isfinite(res.std_errors.to_numpy()))


# --------------------------------------------------------------------------
# 2. Fail loudly
# --------------------------------------------------------------------------


def test_time_without_lag_cutoff_raises_naming_the_missing_arg(geo_panel):
    r = sp.regress("y ~ x1 + x2", data=geo_panel)
    with pytest.raises(ValueError, match="lag_cutoff"):
        sp.conley(r, geo_panel, "lat", "lon", 300.0, time="t", unit="id")


def test_lag_cutoff_without_time_raises(geo_panel):
    r = sp.regress("y ~ x1 + x2", data=geo_panel)
    with pytest.raises(ValueError, match="no time column"):
        sp.conley(r, geo_panel, "lat", "lon", 300.0, lag_cutoff=3, unit="id")


def test_error_message_shows_a_corrected_call(geo_panel):
    r = sp.regress("y ~ x1 + x2", data=geo_panel)
    with pytest.raises(ValueError) as exc:
        sp.conley(r, geo_panel, "lat", "lon", 300.0, time="t", unit="id")
    msg = str(exc.value)
    assert "sp.conley(" in msg and "lag_cutoff=" in msg


def test_time_without_unit_raises(geo_panel):
    r = sp.regress("y ~ x1 + x2", data=geo_panel)
    with pytest.raises(ValueError, match="unit"):
        sp.conley(r, geo_panel, "lat", "lon", 300.0, time="t", lag_cutoff=3)


def test_lag_cutoff_cross_without_time_raises(geo_panel):
    r = sp.regress("y ~ x1 + x2", data=geo_panel)
    with pytest.raises(ValueError, match="lag_cutoff_cross"):
        sp.conley(r, geo_panel, "lat", "lon", 300.0, lag_cutoff_cross=2)


def test_moving_unit_coordinates_raise(geo_panel):
    """A unit must be one fixed place, else the panel KD-tree is a lie."""
    df = geo_panel.copy()
    df.loc[df.index[3], "lat"] += 0.5
    r = sp.regress("y ~ x1 + x2", data=df)
    with pytest.raises(MethodIncompatibility, match="not constant within"):
        sp.conley(r, df, "lat", "lon", 300.0, time="t", lag_cutoff=2, unit="id")


def test_duplicate_unit_time_rows_raise(geo_panel):
    """⚠️ 2026-07: duplicated (unit, time) cells must fail loudly.

    The cross-unit block's cell lookup is single-valued (last-write-wins),
    so before this guard a duplicated cell silently dropped all duplicates
    but one from the cross-unit terms while the within-unit block kept
    every row — wrong SEs with no signal. Realistic trigger: passing a
    coarser geography than the row level (e.g. unit="county" on plant
    rows), which passes the coordinate-constancy check. Stata's acreg
    refuses repeated id-time for the same reason.
    """
    df = pd.concat([geo_panel, geo_panel.iloc[[0]]], ignore_index=True)
    r = sp.regress("y ~ x1 + x2", data=df)
    with pytest.raises(ValueError, match="does not uniquely index"):
        sp.conley(r, df, "lat", "lon", 300.0, time="t", lag_cutoff=2, unit="id")


def test_duplicate_guard_allows_unique_panel(geo_panel):
    """Edge: the guard must not fire on a clean one-row-per-cell panel."""
    r = sp.regress("y ~ x1 + x2", data=geo_panel)
    res = sp.conley(
        r, geo_panel, "lat", "lon", 300.0, time="t", lag_cutoff=2, unit="id"
    )
    assert np.all(np.isfinite(res.std_errors.to_numpy()))


def test_non_integer_time_raises(geo_panel):
    df = geo_panel.copy()
    df["t"] = df["t"] + 0.5
    r = sp.regress("y ~ x1 + x2", data=df)
    with pytest.raises(ValueError, match="integer-valued"):
        sp.conley(r, df, "lat", "lon", 300.0, time="t", lag_cutoff=2, unit="id")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(kernel="triangle"), "kernel must be"),
        (dict(time_kernel="triangle"), "time_kernel must be"),
        (dict(distance="euclidean"), "distance must be"),
    ],
)
def test_unknown_kernel_names_raise(geo_panel, kwargs, match):
    r = sp.regress("y ~ x1 + x2", data=geo_panel)
    with pytest.raises(ValueError, match=match):
        sp.conley(r, geo_panel, "lat", "lon", 300.0, **kwargs)


def test_row_misalignment_raises(geo_panel):
    r = sp.regress("y ~ x1 + x2", data=geo_panel)
    with pytest.raises(MethodIncompatibility, match="row-aligned"):
        sp.conley(r, geo_panel.iloc[:-5], "lat", "lon", 300.0)


# --------------------------------------------------------------------------
# 3. Kernel semantics
# --------------------------------------------------------------------------


def test_lag_cutoff_changes_the_answer(geo_panel):
    """Serial correlation must actually be picked up."""
    r = sp.regress("y ~ x1 + x2", data=geo_panel)
    base = dict(dist_cutoff=300.0, kernel="bartlett", unit="id", time="t")
    se0 = sp.conley(r, geo_panel, "lat", "lon", lag_cutoff=0, **base).std_errors
    se5 = sp.conley(r, geo_panel, "lat", "lon", lag_cutoff=5, **base).std_errors
    assert not np.allclose(se0.to_numpy(), se5.to_numpy(), rtol=1e-8)


def test_cross_unit_lag_defaults_to_zero(geo_panel):
    """Default must be acreg's: lag() governs within-unit pairs only."""
    r = sp.regress("y ~ x1 + x2", data=geo_panel)
    base = dict(dist_cutoff=300.0, kernel="bartlett", unit="id", time="t", lag_cutoff=4)
    implicit = sp.conley(r, geo_panel, "lat", "lon", **base)
    explicit = sp.conley(r, geo_panel, "lat", "lon", lag_cutoff_cross=0, **base)
    np.testing.assert_array_equal(
        implicit.data_info["vcov"], explicit.data_info["vcov"]
    )
    assert implicit.model_info["lag_cutoff_cross"] == 0
    # Opening up the cross-unit time window must change things.
    wide = sp.conley(r, geo_panel, "lat", "lon", lag_cutoff_cross=4, **base)
    assert not np.allclose(
        implicit.std_errors.to_numpy(), wide.std_errors.to_numpy(), rtol=1e-8
    )


def test_time_kernel_uniform_vs_bartlett_differ(geo_panel):
    r = sp.regress("y ~ x1 + x2", data=geo_panel)
    base = dict(dist_cutoff=300.0, kernel="uniform", unit="id", time="t", lag_cutoff=4)
    bart = sp.conley(r, geo_panel, "lat", "lon", time_kernel="bartlett", **base)
    unif = sp.conley(r, geo_panel, "lat", "lon", time_kernel="uniform", **base)
    assert not np.allclose(
        bart.std_errors.to_numpy(), unif.std_errors.to_numpy(), rtol=1e-8
    )


def test_zero_distance_cutoff_collapses_to_within_unit_only(geo_panel):
    """With no spatial reach, only same-unit pairs survive -> cluster-by-unit."""
    r = sp.regress("y ~ x1 + x2", data=geo_panel)
    res = sp.conley(
        r,
        geo_panel,
        "lat",
        "lon",
        dist_cutoff=1e-9,
        kernel="uniform",
        time="t",
        lag_cutoff=geo_panel["t"].max(),
        time_kernel="uniform",
        unit="id",
    )
    X = np.asarray(r.data_info["X"])
    e = np.asarray(r.data_info["residuals"])
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, idx in geo_panel.groupby("id").groups.items():
        pos = geo_panel.index.get_indexer(idx)
        g = (X[pos] * e[pos, None]).sum(axis=0)
        meat += np.outer(g, g)
    np.testing.assert_allclose(
        res.data_info["vcov"], bread @ meat @ bread, rtol=1e-10, atol=0
    )


def test_model_info_records_the_configuration(geo_panel):
    r = sp.regress("y ~ x1 + x2", data=geo_panel)
    res = sp.conley(
        r,
        geo_panel,
        "lat",
        "lon",
        300.0,
        kernel="bartlett",
        time="t",
        lag_cutoff=3,
        unit="id",
        distance="planar",
    )
    mi = res.model_info
    assert mi["se_type"] == "conley_spatiotemporal"
    assert mi["lag_cutoff"] == 3
    assert mi["lag_cutoff_cross"] == 0
    assert mi["kernel"] == "bartlett"
    assert mi["time_kernel"] == "bartlett"
    assert mi["distance"] == "planar"


def test_vcov_is_symmetric(geo_panel):
    r = sp.regress("y ~ x1 + x2", data=geo_panel)
    for dist_mode in ("haversine", "planar"):
        V = sp.conley(
            r,
            geo_panel,
            "lat",
            "lon",
            300.0,
            kernel="bartlett",
            time="t",
            lag_cutoff=3,
            unit="id",
            distance=dist_mode,
        ).data_info["vcov"]
        np.testing.assert_allclose(V, V.T, rtol=0, atol=1e-18)


def test_planar_vcov_is_variable_order_invariant(geo_panel):
    """Unlike acreg, reordering the regressors must not move any entry."""
    r12 = sp.regress("y ~ x1 + x2", data=geo_panel)
    r21 = sp.regress("y ~ x2 + x1", data=geo_panel)
    kw = dict(
        dist_cutoff=300.0,
        kernel="bartlett",
        time="t",
        lag_cutoff=3,
        unit="id",
        distance="planar",
    )
    v12 = sp.conley(r12, geo_panel, "lat", "lon", **kw)
    v21 = sp.conley(r21, geo_panel, "lat", "lon", **kw)
    got = v12.std_errors.reindex(["Intercept", "x1", "x2"]).to_numpy()
    want = v21.std_errors.reindex(["Intercept", "x1", "x2"]).to_numpy()
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)


# --------------------------------------------------------------------------
# 4. Memory / scaling
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_panel_path_does_not_allocate_n_squared():
    """250 units x 100 periods = 25,000 rows.

    A dense n x n float64 weight matrix would be 25,000^2 * 8 = 5.0 GB. The
    de-duplicated KD-tree plus block accumulation must stay orders of
    magnitude below that.
    """
    rng = np.random.default_rng(5)
    n_units, n_periods = 250, 100
    lat0 = rng.uniform(35.0, 45.0, n_units)
    lon0 = rng.uniform(-115.0, -100.0, n_units)
    df = pd.DataFrame(
        [(u, t, lat0[u], lon0[u]) for u in range(n_units) for t in range(n_periods)],
        columns=["id", "t", "lat", "lon"],
    )
    n = len(df)
    df["x1"] = rng.normal(size=n)
    df["x2"] = rng.normal(size=n)
    df["y"] = 1.0 + 0.5 * df.x1 - 0.3 * df.x2 + rng.normal(size=n)

    r = sp.regress("y ~ x1 + x2", data=df)

    dense_bytes = n * n * 8
    assert dense_bytes > 4e9, "fixture too small to be a meaningful guard"

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        res = sp.conley(
            r,
            df,
            "lat",
            "lon",
            dist_cutoff=300.0,
            kernel="bartlett",
            time="t",
            lag_cutoff=10,
            unit="id",
            distance="planar",
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert np.all(np.isfinite(res.std_errors.to_numpy()))
    # Generous ceiling: still ~25x smaller than the dense weight matrix.
    assert peak < 200e6, f"peak allocation {peak / 1e6:.0f} MB is too high"
    assert peak < dense_bytes / 25


@pytest.mark.slow
def test_kdtree_is_built_on_units_not_rows(geo_panel, monkeypatch):
    """Panel de-duplication: the tree must see 40 units, not 600 rows."""
    import sys

    # `statspai.inference.conley` resolves to the re-exported *function*, so
    # reach for the module object explicitly.
    mod = sys.modules["statspai.inference.conley"]

    sizes = []
    real = mod.cKDTree

    def spy(data, *a, **kw):
        sizes.append(len(data))
        return real(data, *a, **kw)

    monkeypatch.setattr(mod, "cKDTree", spy)
    r = sp.regress("y ~ x1 + x2", data=geo_panel)
    sp.conley(
        r,
        geo_panel,
        "lat",
        "lon",
        300.0,
        kernel="bartlett",
        time="t",
        lag_cutoff=3,
        unit="id",
    )
    assert sizes, "cKDTree was never constructed"
    assert all(s == geo_panel["id"].nunique() for s in sizes), sizes
    assert all(s < len(geo_panel) for s in sizes)
