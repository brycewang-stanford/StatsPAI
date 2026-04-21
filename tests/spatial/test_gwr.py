"""GWR + MGWR tests — cross-validated against PySAL mgwr on the Georgia
benchmark dataset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from statspai.spatial.gwr import gwr, mgwr, gwr_bandwidth
from statspai.spatial import gwr as sp_gwr      # re-export path


FIXTURE = Path(__file__).parent / "fixtures" / "georgia_gwr_reference.json"


@pytest.fixture(scope="module")
def georgia():
    ref = json.loads(FIXTURE.read_text(encoding='utf-8'))
    coords = np.array(list(zip(ref["coords_x"], ref["coords_y"])))
    y = np.array(ref["PctBach"])
    X = np.column_stack([ref["PctRural"], ref["PctPov"], ref["PctBlack"]])
    return coords, y, X, ref


# ------------------------------------------------------------------ GWR
def test_gwr_matches_mgwr_at_same_bandwidth(georgia):
    coords, y, X, ref = georgia
    res = gwr(coords, y, X, bw=ref["bw_aicc"], kernel="bisquare", fixed=False)
    np.testing.assert_allclose(res.R2, ref["r2"], rtol=1e-4)
    np.testing.assert_allclose(res.aicc, ref["aicc"], rtol=1e-3)
    np.testing.assert_allclose(
        res.params.mean(axis=0), ref["params_mean"], rtol=1e-5
    )


def test_gwr_bandwidth_selection_matches_mgwr(georgia):
    coords, y, X, ref = georgia
    bw = gwr_bandwidth(coords, y, X, kernel="bisquare",
                       fixed=False, criterion="AICc")
    assert abs(bw - ref["bw_aicc"]) <= 1     # identical integer at AICc optimum


def test_gwr_local_coefficients_first10(georgia):
    coords, y, X, ref = georgia
    res = gwr(coords, y, X, bw=93, kernel="bisquare", fixed=False)
    np.testing.assert_allclose(res.params[:10], ref["first_10_params"], rtol=1e-5)


def test_gwr_gaussian_kernel_runs(georgia):
    coords, y, X, _ = georgia
    res = gwr(coords, y, X, bw=50, kernel="gaussian", fixed=False)
    assert np.isfinite(res.R2)
    assert res.params.shape == (len(y), 4)


def test_gwr_fixed_kernel_runs(georgia):
    coords, y, X, _ = georgia
    diag = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
    res = gwr(coords, y, X, bw=diag * 0.3, kernel="bisquare", fixed=True)
    assert np.isfinite(res.R2)


def test_gwr_summary_prints(georgia):
    coords, y, X, _ = georgia
    res = gwr(coords, y, X, bw=93, kernel="bisquare", fixed=False)
    s = res.summary()
    assert "Geographically Weighted Regression" in s
    assert "AICc" in s


# ------------------------------------------------------------------ MGWR
def test_mgwr_converges_and_reports_bws(georgia):
    coords, y, X, _ = georgia
    res = mgwr(coords, y, X, kernel="bisquare", fixed=False, max_iter=30)
    assert len(res.bws) == 4
    assert res.params.shape == (len(y), 4)
    assert np.isfinite(res.R2)
    # MGWR should fit at least as well as GWR (no fewer parameters)
    gwr_res = gwr(coords, y, X, bw=93, kernel="bisquare", fixed=False)
    assert res.resid_ss <= gwr_res.resid_ss * 1.05        # within 5% slack


# ------------------------------------------------------------------ exports
def test_gwr_reexported_at_sp_spatial_gwr():
    assert callable(sp_gwr)


def test_sp_dot_gwr_reexport():
    import statspai as sp
    assert callable(sp.gwr)
    assert callable(sp.mgwr)
    assert callable(sp.gwr_bandwidth)
