"""Write the frontier / structural parity datasets (fixed seeds).

Run from anywhere::

    python tests/reference_parity/_generate_frontier_struct_data.py

Writes into ``tests/reference_parity/_fixtures/``:

* ``frontier_struct_lcsf.csv``  -- 600 obs, two technology classes (for
  ``sp.lcsf`` vs ``sfaR::sfalcmcross``) with a class-probability shifter z.
* ``frontier_struct_zisf.csv``  -- 500 obs, 40 % fully efficient firms (for
  ``sp.zisf``; no reference implementation, identity checks only).
* ``frontier_struct_meta.csv``  -- 3 groups x 90 obs, Cobb-Douglas in two
  inputs with group-specific slopes so the group frontiers cross inside the
  data (for ``sp.metafrontier`` vs ``metafrontier::metafrontier``).
* ``frontier_struct_malm.csv``  -- balanced panel, 80 firms x 3 periods (for
  ``sp.malmquist``).

Values are written with 17 significant digits so R, Stata and Python read
identical doubles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "_fixtures"


def _write(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False, float_format="%.17g")


def lcsf_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260918)
    n = 600
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    z = rng.normal(0.0, 1.0, n)
    p1 = 1.0 / (1.0 + np.exp(-(0.3 + 0.8 * z)))
    cls1 = rng.uniform(size=n) < p1
    v = rng.normal(0.0, 0.15, n)
    u1 = np.abs(rng.normal(0.0, 0.20, n))
    u2 = np.abs(rng.normal(0.0, 0.60, n))
    y = np.where(
        cls1,
        1.0 + 0.6 * x1 + 0.3 * x2 + v - u1,
        0.4 + 0.3 * x1 + 0.6 * x2 + v - u2,
    )
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "z": z})


def zisf_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260919)
    n = 500
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    eff = rng.uniform(size=n) < 0.4
    v = rng.normal(0.0, 0.15, n)
    u = np.where(eff, 0.0, np.abs(rng.normal(0.0, 0.5, n)))
    y = 1.0 + 0.6 * x1 + 0.3 * x2 + v - u
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def meta_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260920)
    rows = []
    spec = {"A": (1.0, 0.5, 0.4), "B": (1.2, 0.3, 0.6), "C": (0.8, 0.7, 0.2)}
    for g, (b0, b1, b2) in spec.items():
        n = 90
        x1 = rng.normal(0.0, 1.0, n)
        x2 = rng.normal(0.0, 1.0, n)
        v = rng.normal(0.0, 0.15, n)
        u = np.abs(rng.normal(0.0, 0.35, n))
        y = b0 + b1 * x1 + b2 * x2 + v - u
        rows.append(pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": g}))
    return pd.concat(rows, ignore_index=True)


def malm_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260921)
    n_firm, periods = 80, (1, 2, 3)
    x1_0 = rng.normal(0.0, 1.0, n_firm)
    x2_0 = rng.normal(0.0, 1.0, n_firm)
    rows = []
    for t in periods:
        x1 = x1_0 + rng.normal(0.05 * t, 0.2, n_firm)
        x2 = x2_0 + rng.normal(0.0, 0.2, n_firm)
        v = rng.normal(0.0, 0.12, n_firm)
        u = np.abs(rng.normal(0.0, 0.35, n_firm))
        y = (1.0 + 0.08 * t) + (0.5 + 0.02 * t) * x1 + 0.3 * x2 + v - u
        rows.append(
            pd.DataFrame(
                {"id": np.arange(1, n_firm + 1), "t": t, "y": y, "x1": x1, "x2": x2}
            )
        )
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    _write(lcsf_data(), "frontier_struct_lcsf.csv")
    _write(zisf_data(), "frontier_struct_zisf.csv")
    _write(meta_data(), "frontier_struct_meta.csv")
    _write(malm_data(), "frontier_struct_malm.csv")
