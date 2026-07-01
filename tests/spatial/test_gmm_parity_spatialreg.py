"""Spatial GMM estimators must stay pinned to spatialreg.

Regression lock for the module-66 parity rows: ``sp.sar_gmm`` (Kelejian-Prucha
spatial 2SLS) reproduces ``spatialreg::stsls(W2X=FALSE)`` — a closed-form
projection, so coefficients *and* the ``sig2n_k`` SEs agree to machine
precision — and ``sp.sem_gmm`` reproduces ``spatialreg::GMerrorsar`` on the
coefficients and the spatial-error parameter ``lambda`` (the SEs use a
different residual-variance convention and are not compared here).

Reference values are the committed R goldens from
``tests/r_parity/66_spatial_gmm.R`` (spatialreg 1.4.3 / spdep 1.4.2). This test
regenerates the identical rook-lattice DGP and refits the Python estimators so
a silent regression fails here, not only when the parity harness is rebuilt.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai.spatial import sar_gmm, sem_gmm

_R_GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "r_parity"
    / "results"
    / "66_spatial_gmm_R.json"
)


def _make_dgp(side: int = 12, seed: int = 43):
    """Rebuild the module-66 rook-lattice spatial-error DGP (66_spatial_gmm.py)."""
    rng = np.random.default_rng(seed)
    n = side * side
    grid_row = np.repeat(np.arange(side), side)
    grid_col = np.tile(np.arange(side), side)
    dr = np.abs(grid_row[:, None] - grid_row[None, :])
    dc = np.abs(grid_col[:, None] - grid_col[None, :])
    W = ((dr + dc) == 1).astype(float)
    Wn = W / W.sum(1, keepdims=True)

    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    u = np.linalg.solve(np.eye(n) - 0.5 * Wn, rng.normal(size=n))
    y = X @ np.array([1.0, 0.7, -0.4]) + u
    return W, pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _r_reference():
    rows = json.loads(_R_GOLDEN.read_text(encoding="utf-8"))["rows"]
    return {r["statistic"]: r for r in rows}


def test_sar_gmm_matches_stsls_point_and_se():
    W, df = _make_dgp()
    fit = sar_gmm(W, df, "y ~ x1 + x2", w_lags=1)
    ref = _r_reference()
    for key in ["const", "x1", "x2", "rho"]:
        r = ref[f"sar_gmm_{key}"]
        # Closed-form 2SLS projection: machine-level agreement on both.
        assert float(fit.params[key]) == pytest.approx(r["estimate"], rel=1e-6), key
        assert float(fit.std_errors[key]) == pytest.approx(r["se"], rel=1e-6), key


def test_sem_gmm_matches_gmerrorsar_coefficients():
    W, df = _make_dgp()
    fit = sem_gmm(W, df, "y ~ x1 + x2")
    ref = _r_reference()
    for key in ["const", "x1", "x2", "lambda"]:
        r = ref[f"sem_gmm_{key}"]
        assert float(fit.params[key]) == pytest.approx(r["estimate"], rel=1e-5), key
