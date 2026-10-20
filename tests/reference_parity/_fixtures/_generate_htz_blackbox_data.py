"""Designs for the CR2 / HTZ Wald-test black-box fixture.

Writes ``htz_blackbox_designs.csv`` (all designs stacked; column ``design``)
and ``htz_blackbox_tests.json`` (restriction matrices and null values per
design). ``_generate_htz_blackbox_R.R`` runs R ``clubSandwich`` on them.

Designs: (0) balanced, (1) unbalanced with singleton clusters, (2) a
cluster-level regressor, (3) a balanced unit-by-period panel with unit and
period dummies, clustered by unit, where ``I - H_gg`` is singular (the
Moore-Penrose branch of the CR2 adjustment), (4) few clusters, heavy
leverage.

    python tests/reference_parity/_fixtures/_generate_htz_blackbox_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def _design(d: int, rng: np.random.Generator) -> pd.DataFrame:
    if d == 3:
        units, periods = 12, 6
        g = np.repeat(np.arange(units), periods)
        t = np.tile(np.arange(periods), units)
        x1 = rng.normal(size=g.size) + 0.5 * rng.normal(size=units)[g]
        x2 = rng.normal(size=g.size)
        cols = {"g": g, "x1": x1, "x2": x2}
        for u in range(1, units):
            cols[f"u{u}"] = (g == u).astype(float)
        for p in range(1, periods):
            cols[f"t{p}"] = (t == p).astype(float)
        df = pd.DataFrame(cols)
        df["y"] = (
            0.4 * x1 - 0.3 * x2 + rng.normal(size=units)[g] + rng.normal(size=g.size)
        )
        return df
    if d == 0:
        sizes = np.full(18, 12)
    elif d == 1:
        sizes = rng.integers(1, 25, size=30)
        sizes[[0, 7]] = 1
    elif d == 2:
        sizes = rng.integers(5, 20, size=22)
    else:
        sizes = rng.integers(3, 40, size=9)
    G = sizes.size
    g = np.repeat(np.arange(G), sizes)
    x1 = rng.normal(size=g.size) + rng.normal(size=G)[g]
    x2 = rng.normal(size=g.size)
    x3 = rng.normal(size=G)[g] if d == 2 else rng.standard_t(3, size=g.size)
    if d == 4:
        x1[0] = 8.0
    y = 0.3 * x1 - 0.2 * x2 + 0.1 * x3 + rng.normal(size=G)[g] + rng.normal(size=g.size)
    return pd.DataFrame({"g": g, "x1": x1, "x2": x2, "x3": x3, "y": y})


def main() -> None:
    rng = np.random.default_rng(20261020)
    frames, tests = [], {}
    for d in range(5):
        df = _design(d, rng)
        df.insert(0, "design", d)
        frames.append(df)
        k = 4 if d != 3 else 3 + 11 + 5  # intercept, x's, (dummies)
        cases = []
        # Coefficients that vary within clusters: x1, x2 and x3 (or, in the
        # panel, the first period dummy; a unit dummy is not identified by
        # the cluster-robust variance).
        tested = [1, 2, 3] if d != 3 else [1, 2, 14]
        for q in (1, 2, 3):
            R = np.zeros((q, k))
            for s in range(q):
                R[s, tested[s]] = 1.0
            cases.append({"R": R.tolist(), "r": [0.0] * q})
        mix = rng.normal(size=(2, 3))
        R = np.zeros((2, k))
        R[:, tested] = mix
        cases.append({"R": R.tolist(), "r": [0.05, -0.1]})
        if d != 3:
            cases.append({"R": np.eye(k).tolist(), "r": [0.0] * k})
        tests[str(d)] = cases
    pd.concat(frames, ignore_index=True).to_csv(
        HERE / "htz_blackbox_designs.csv", index=False, float_format="%.17g"
    )
    (HERE / "htz_blackbox_tests.json").write_text(
        json.dumps(tests, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
