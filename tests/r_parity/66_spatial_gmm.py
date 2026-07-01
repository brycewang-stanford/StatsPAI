"""StatsPAI spatial GMM (SAR-2SLS / SEM-GMM) parity — Module 66 (Python side).

Generates a deterministic 12x12 rook-lattice spatial-error DGP and fits the
two moment-based spatial estimators whose conventions align with
``spatialreg``:

  * ``sp.sar_gmm(..., w_lags=1)`` — Kelejian-Prucha spatial two-stage least
    squares with instruments ``[X, WX]`` and endogenous ``WY``. Matches
    ``spatialreg::stsls(..., W2X = FALSE)`` on both coefficients and the
    ``sig2n_k`` (n-k) standard errors.
  * ``sp.sem_gmm(...)`` — Kelejian-Prucha (1999) generalized-moments spatial
    error model. Matches ``spatialreg::GMerrorsar`` on the coefficients and
    the spatial-error parameter ``lambda``. Emitted point-only: the two
    implementations use different residual-variance estimators for the
    coefficient SEs (~1.5% apart) and sp does not return a lambda SE, so the
    SE rows are not a like-for-like join.

``sp.sarar_gmm`` is deliberately *not* included: its joint feasible-GS lag +
GM-error path does not match ``spatialreg::gstsls`` (different moment
sequence), so no honest parity row exists yet.

Both sides rebuild the identical rook contiguity from the grid coordinates,
row-standardise it (spdep style = "W"), and see byte-identical W.

Tolerance: rel_est < 1e-6 (machine). The spatial 2SLS estimator is a
closed-form projection given the instruments, so it agrees to machine
tolerance; the GM error coefficients likewise land on the same root.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "66_spatial_gmm"
SIDE = 12  # 12 x 12 lattice -> n = 144


def _rook_W(grid_row: np.ndarray, grid_col: np.ndarray) -> np.ndarray:
    """Binary rook contiguity from integer grid coordinates."""
    dr = np.abs(grid_row[:, None] - grid_row[None, :])
    dc = np.abs(grid_col[:, None] - grid_col[None, :])
    return ((dr + dc) == 1).astype(float)


def make_data(seed: int = PARITY_SEED + 1) -> pd.DataFrame:
    """Rook-lattice spatial-error DGP: y = Xb + u, u = (I - 0.5 W)^-1 eps."""
    rng = np.random.default_rng(seed)
    n = SIDE * SIDE
    grid_row = np.repeat(np.arange(SIDE), SIDE)
    grid_col = np.tile(np.arange(SIDE), SIDE)
    W = _rook_W(grid_row, grid_col)
    Wn = W / W.sum(1, keepdims=True)

    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    eps = rng.normal(size=n)
    u = np.linalg.solve(np.eye(n) - 0.5 * Wn, eps)
    y = X @ np.array([1.0, 0.7, -0.4]) + u
    return pd.DataFrame(
        {"y": y, "x1": x1, "x2": x2, "grid_row": grid_row, "grid_col": grid_col}
    )


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    W = _rook_W(df["grid_row"].to_numpy(), df["grid_col"].to_numpy())
    n = len(df)
    rows: list[ParityRecord] = []

    # SAR 2SLS (w_lags=1 == stsls W2X=FALSE): point + sig2n_k SE, bit-exact.
    res_sar = sp.sar_gmm(W, data=df, formula="y ~ x1 + x2", w_lags=1)
    for name in ["const", "x1", "x2", "rho"]:
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"sar_gmm_{name}",
                estimate=float(res_sar.params[name]),
                se=float(res_sar.std_errors[name]),
                n=n,
            )
        )

    # SEM GMM (GMerrorsar): coefficients + lambda bit-exact; point-only
    # because the coefficient-SE variance estimators differ by convention.
    res_sem = sp.sem_gmm(W, data=df, formula="y ~ x1 + x2")
    for name in ["const", "x1", "x2", "lambda"]:
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"sem_gmm_{name}",
                estimate=float(res_sem.params[name]),
                se=None,
                n=n,
            )
        )

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "n": n,
            "lattice": f"{SIDE}x{SIDE} rook contiguity, row-standardised (style='W')",
            "note": (
                "sp.sar_gmm(w_lags=1) vs spatialreg::stsls(W2X=FALSE) "
                "(coefficients + n-k SEs, bit-exact); sp.sem_gmm vs "
                "spatialreg::GMerrorsar (coefficients + lambda, point-only — "
                "SE variance-estimator convention differs)."
            ),
        },
    )


if __name__ == "__main__":
    main()
