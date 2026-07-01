"""StatsPAI spatial ML (SAR / SEM / SDM) parity — Module 65 (Python side).

Generates a deterministic 12x12 rook-lattice DGP (a spatial-lag process
with row-standardised contiguity) and fits the three maximum-likelihood
spatial regressions. The companion 65_spatial.R runs the canonical R
reference — ``spatialreg::lagsarlm`` (SAR), ``spatialreg::errorsarlm``
(SEM), and ``spatialreg::lagsarlm(..., Durbin = TRUE)`` (SDM) — on the
same CSV, rebuilding the identical rook-contiguity weights from the grid
coordinates so both sides see byte-identical W.

Both sides emit every coefficient with its asymptotic standard error, so
``compare.py`` grades the point estimates *and* the full-information SEs.
sp.sar / sp.sdm read Var(beta) from the leading block of the inverted
(beta, rho, sigma2) information matrix — the same covariance
spatialreg reports — rather than the naive concentrated sigma2 (X'X)^-1.

Tolerance: rel_est < 1e-6, rel_se < 1e-6 (machine tier). The bounded rho /
lambda ML line-search is tightened to xatol=1e-10, so the worst observed gap
vs spatialreg is ~8.3e-8 on estimates and ~2.0e-8 on standard errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "65_spatial"
SIDE = 12  # 12 x 12 lattice -> n = 144


def _rook_W(grid_row: np.ndarray, grid_col: np.ndarray) -> np.ndarray:
    """Binary rook contiguity from integer grid coordinates."""
    dr = np.abs(grid_row[:, None] - grid_row[None, :])
    dc = np.abs(grid_col[:, None] - grid_col[None, :])
    return ((dr + dc) == 1).astype(float)


def make_data(seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = SIDE * SIDE
    grid_row = np.repeat(np.arange(SIDE), SIDE)
    grid_col = np.tile(np.arange(SIDE), SIDE)
    W = _rook_W(grid_row, grid_col)
    Wn = W / W.sum(1, keepdims=True)

    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    beta = np.array([1.0, 0.7, -0.4])
    rho = 0.5
    X = np.column_stack([np.ones(n), x1, x2])
    eps = rng.normal(scale=1.0, size=n)
    y = np.linalg.solve(np.eye(n) - rho * Wn, X @ beta + eps)
    return pd.DataFrame(
        {"y": y, "x1": x1, "x2": x2, "grid_row": grid_row, "grid_col": grid_col}
    )


def _coef_rows(model: str, res, names, n: int) -> list[ParityRecord]:
    rows: list[ParityRecord] = []
    for name in names:
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"{model}_{name}",
                estimate=float(res.params[name]),
                se=float(res.std_errors[name]),
                n=n,
            )
        )
    return rows


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    W = _rook_W(df["grid_row"].to_numpy(), df["grid_col"].to_numpy())
    n = len(df)

    rows: list[ParityRecord] = []

    res_sar = sp.sar(W, data=df, formula="y ~ x1 + x2")
    rows += _coef_rows("sar", res_sar, ["const", "x1", "x2", "rho"], n)

    res_sem = sp.sem(W, data=df, formula="y ~ x1 + x2")
    rows += _coef_rows("sem", res_sem, ["const", "x1", "x2", "lambda"], n)

    res_sdm = sp.sdm(W, data=df, formula="y ~ x1 + x2")
    rows += _coef_rows("sdm", res_sdm, ["const", "x1", "x2", "W_x1", "W_x2", "rho"], n)

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "n": n,
            "lattice": f"{SIDE}x{SIDE} rook contiguity, row-standardised (style='W')",
            "note": (
                "sp.sar/sem/sdm ML fits vs spatialreg::lagsarlm / errorsarlm / "
                "lagsarlm(Durbin=TRUE). Coefficients and full-information "
                "asymptotic SEs match to machine tolerance."
            ),
        },
    )


if __name__ == "__main__":
    main()
