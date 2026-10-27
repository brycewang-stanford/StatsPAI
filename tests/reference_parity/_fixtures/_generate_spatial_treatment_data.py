"""Spatial-treatment designs for the ``sp.spatial_treatment`` parity check.

Two designs on the unit square, stacked in ``spatial_treatment_units.csv``
and ``spatial_treatment_locations.csv`` (column ``design``). Candidate
locations sit on a jittered grid; outcome units are uniform. Design 1 has a
constant design probability, design 2 heterogeneous probabilities. Effects
are additive (2 within 0.05 of each realised location, 0.5 within 0.10).
``_generate_spatial_treatment_R.R`` runs the author's reference code
(Pollmann, spatialTreat-example, MIT licence) on them.

    python tests/reference_parity/_fixtures/_generate_spatial_treatment_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def main() -> None:
    rng = np.random.default_rng(20261028)
    units, locs = [], []
    for design in (1, 2):
        g = (np.arange(7) + 0.5) / 7
        lx = np.repeat(g, 7) + rng.uniform(-0.02, 0.02, 49)
        ly = np.tile(g, 7) + rng.uniform(-0.02, 0.02, 49)
        p = np.full(49, 0.4) if design == 1 else rng.uniform(0.2, 0.7, 49)
        W = (rng.uniform(size=49) < p).astype(int)
        ux, uy = rng.uniform(0, 1, 500), rng.uniform(0, 1, 500)
        d = np.hypot(ux[:, None] - lx[None, :], uy[:, None] - ly[None, :])
        base = 10 + 3 * np.sin(4 * ux) * np.cos(3 * uy) + rng.normal(size=500)
        eff = ((d <= 0.05) * 2.0 + ((d > 0.05) & (d <= 0.10)) * 0.5) @ W
        units.append(
            pd.DataFrame(
                {
                    "design": design,
                    "i": np.arange(500),
                    "x": ux,
                    "y": uy,
                    "Y": base + eff,
                }
            )
        )
        locs.append(
            pd.DataFrame(
                {"design": design, "s": np.arange(49), "x": lx, "y": ly, "p": p, "W": W}
            )
        )
    pd.concat(units).to_csv(
        HERE / "spatial_treatment_units.csv", index=False, float_format="%.17g"
    )
    pd.concat(locs).to_csv(
        HERE / "spatial_treatment_locations.csv", index=False, float_format="%.17g"
    )


if __name__ == "__main__":
    main()
