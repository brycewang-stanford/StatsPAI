"""Shift-share designs whose AKM0 confidence sets take every shape.

Six designs of 120 locations and 12 sectors, stacked in
``akm_ci_shapes_loc.csv`` (column ``design``) with their shares in
``akm_ci_shapes_shares.csv``: strong, moderate and weak first stages, so the
null-imposed AKM0 set is a bounded interval, the union of two half-lines or
the whole real line; designs 4-5 carry location weights and design 5 an
exactly duplicated share column. R ``ShiftShareSE`` is run on them as a
black box by ``_generate_akm_ci_shapes_R.R``.

    python tests/reference_parity/_fixtures/_generate_akm_ci_shapes_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
N, K = 120, 12
STRENGTH = [1.0, 0.25, 0.04, 0.0, 0.5, 0.08]


def main() -> None:
    rng = np.random.default_rng(20261020)
    rows, shares = [], []
    for d, pi in enumerate(STRENGTH):
        S = rng.dirichlet(np.full(K, 0.6), size=N) * rng.uniform(0.6, 1.0, (N, 1))
        if d == 5:
            S[:, K - 1] = S[:, 2]
        g = rng.normal(size=K)
        X = S @ g
        c1 = rng.normal(size=N)
        u = rng.normal(size=N)
        x = pi * X + 0.4 * c1 + u
        y = 0.7 * x - 0.3 * c1 + 0.8 * u + rng.normal(size=N)
        w = rng.lognormal(sigma=0.5, size=N) if d >= 4 else np.ones(N)
        region = rng.integers(1, 16, size=N)
        rows.append(
            pd.DataFrame(
                {
                    "design": d,
                    "y": y,
                    "x": x,
                    "X": X,
                    "c1": c1,
                    "w": w,
                    "region": region,
                }
            )
        )
        shares.append(S)
    pd.concat(rows, ignore_index=True).to_csv(
        HERE / "akm_ci_shapes_loc.csv", index=False, float_format="%.17g"
    )
    pd.DataFrame(np.vstack(shares)).to_csv(
        HERE / "akm_ci_shapes_shares.csv",
        index=False,
        header=False,
        float_format="%.17g",
    )


if __name__ == "__main__":
    main()
