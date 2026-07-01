"""SAR/SDM coefficient SEs must use the full-information covariance.

Regression lock for the spatial-ML SE fix: ``sp.sar`` / ``sp.sdm`` read
``Var(beta)`` from the leading block of the inverted ``(beta, rho, sigma2)``
information matrix — exactly the asymptotic covariance
``spatialreg::lagsarlm`` reports — rather than the naive concentrated
``sigma2 (X'X)^-1`` (which treats ``rho`` as known and understates the SE,
roughly halving the intercept SE on a row-standardised ``W``).

The reference values are the committed R goldens produced by
``tests/r_parity/65_spatial.R`` (spatialreg 1.4.3 / spdep 1.4.2); this test
regenerates the identical 12x12 rook-lattice DGP, refits the Python
estimators, and requires machine-level agreement — so any silent regression
back to the concentrated SE fails here, not only when the parity harness is
re-run by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai.spatial import sar, sdm

_R_GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "r_parity"
    / "results"
    / "65_spatial_R.json"
)


def _make_dgp(side: int = 12, seed: int = 42):
    """Rebuild the module-65 rook-lattice SAR DGP (identical to 65_spatial.py)."""
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
    y = np.linalg.solve(
        np.eye(n) - 0.5 * Wn, X @ np.array([1.0, 0.7, -0.4]) + rng.normal(size=n)
    )
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    return W, df


def _r_reference():
    rows = json.loads(_R_GOLDEN.read_text(encoding="utf-8"))["rows"]
    return {r["statistic"]: r for r in rows}


@pytest.mark.parametrize(
    "model, keys",
    [
        ("sar", ["const", "x1", "x2", "rho"]),
        ("sdm", ["const", "x1", "x2", "W_x1", "W_x2", "rho"]),
    ],
)
def test_spatial_ml_se_matches_spatialreg(model, keys):
    W, df = _make_dgp()
    fit = {"sar": sar, "sdm": sdm}[model](W, df, "y ~ x1 + x2")
    ref = _r_reference()
    for key in keys:
        r = ref[f"{model}_{key}"]
        assert float(fit.params[key]) == pytest.approx(r["estimate"], rel=1e-5), key
        assert float(fit.std_errors[key]) == pytest.approx(r["se"], rel=1e-4), key


def test_sar_intercept_se_beats_concentrated_formula():
    """The full-information intercept SE must exceed the naive concentrated
    ``sigma2 (X'X)^-1`` value — the concrete symptom of the bug that was fixed."""
    W, df = _make_dgp()
    fit = sar(W, df, "y ~ x1 + x2")

    y = df["y"].to_numpy()
    n = len(df)
    X = np.column_stack([np.ones(n), df["x1"], df["x2"]])
    Wn = W / W.sum(1, keepdims=True)
    rho = float(fit.params["rho"])
    beta = fit.params[["const", "x1", "x2"]].to_numpy()
    e = (y - rho * (Wn @ y)) - X @ beta
    sigma2 = float(e @ e) / n
    naive_se_const = float(np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X))[0]))

    # Full-information intercept SE is materially larger (here ~2x) than the
    # concentrated one that ignores the rho-beta covariance.
    assert float(fit.std_errors["const"]) > 1.5 * naive_se_const
