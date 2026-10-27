"""Cross-section of coastal grid cells for the IV reporting tools.

Shaped after Caprettini, Trew and Voth's design (one row per cell, an
instrument that varies smoothly in space, spatially correlated errors, a
handful of regions): 600 cells on a jittered grid over 50-56 N, 6 W - 2 E.

* ``z``  instrument, a smooth function of location plus noise; ``z2`` a
  second, weak instrument (over-identified weak-IV tests)
* ``x``  endogenous regressor, ``v`` its first-stage error is correlated
  with the outcome error
* ``w1``, ``w2`` controls (``w1`` in different units on purpose), ``d`` a dummy
* ``u``  spatially correlated error (a smooth field plus a correlated
  neighbourhood shock), heteroskedastic in ``w1``
* ``region`` 12 longitude-latitude blocks (Ibragimov-Mueller groups),
  ``fe`` 30 fixed-effect groups, ``wt`` analytic weights

All floats are rounded to 6 decimals so that Stata's ``import delimited,
asdouble`` and pandas read the same doubles.

    python tests/reference_parity/_fixtures/_generate_ivreport_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "ivreport_fx.csv"


def main() -> None:
    rng = np.random.default_rng(20261028)
    n = 600
    lat = 50.0 + 6.0 * rng.uniform(size=n)
    lon = -6.0 + 8.0 * rng.uniform(size=n)
    field = np.sin(lat * 1.3) + np.cos(lon * 0.9)
    # correlated neighbourhood shock: average of nearby iid draws
    eps = rng.normal(size=n)
    d_lat = 111.0 * (lat[:, None] - lat[None, :])
    d_lon = 111.0 * np.cos(np.radians(lat))[None, :] * (lon[:, None] - lon[None, :])
    near = np.sqrt(d_lat**2 + d_lon**2) < 60.0
    shock = (near @ eps) / near.sum(axis=1)
    z = 0.8 * np.cos(lat * 0.7) + 0.5 * rng.normal(size=n)
    z2 = 0.5 * np.sin(lon * 0.8) + rng.normal(size=n)
    w1 = 10.0 + 4.0 * rng.normal(size=n)
    w2 = rng.normal(size=n)
    d = (rng.uniform(size=n) < 0.35).astype(float)
    v = rng.normal(size=n)
    x = 0.6 * z + 0.12 * z2 + 0.05 * w1 + 0.3 * field + v
    u = 0.6 * field + 1.2 * shock + (0.5 + 0.05 * np.abs(w1 - 10)) * rng.normal(size=n)
    y = 1.0 + 0.4 * x + 0.08 * w1 - 0.3 * w2 + 0.2 * d + u + 0.7 * v
    lat_band = np.minimum((lat - 50.0) // 2.0, 2).astype(int)
    lon_band = np.minimum((lon + 6.0) // 2.0, 3).astype(int)
    region = 1 + lat_band * 4 + lon_band
    fe = 1 + rng.integers(0, 30, size=n)
    wt = rng.uniform(0.5, 2.0, size=n)
    out = pd.DataFrame(
        {
            "y": y,
            "x": x,
            "z": z,
            "z2": z2,
            "w1": w1,
            "w2": w2,
            "d": d,
            "lat": lat,
            "lon": lon,
            "region": region,
            "fe": fe,
            "wt": wt,
        }
    ).round(6)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
