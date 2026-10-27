"""Data for the Sanderson-Windmeijer first-stage fixture (``iv_sw.csv``).

600 observations, three endogenous regressors and four instruments, two
crossed fixed effects (``f`` with 30 levels, ``g`` with 40), a cluster
variable ``c`` (25 clusters) crossing both, and ``fc = f // 3`` in which
``f`` is nested. ``z4`` moves only ``d3`` and ``d1`` and ``d2`` load on
nearly the same instrument direction, so the conditional (SW) statistics
differ clearly from the plain first-stage F.

    python tests/reference_parity/_fixtures/_generate_iv_sw_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "iv_sw.csv"


def main() -> None:
    rng = np.random.default_rng(20261028)
    n = 600
    f = rng.integers(0, 30, n)
    g = rng.integers(0, 40, n)
    c = rng.integers(0, 25, n)
    Z = rng.normal(size=(n, 4)) + rng.normal(size=25)[c][:, None] * 0.3
    w = rng.normal(size=n) + 0.2 * f / 30
    u = rng.normal(size=n) * (1 + 0.5 * np.abs(Z[:, 0]))
    fe = rng.normal(size=30)[f] + rng.normal(size=40)[g]
    d1 = Z @ [1.0, 0.6, 0.1, 0.0] + 0.5 * u + fe + rng.normal(size=n)
    d2 = Z @ [0.8, 0.5, 0.2, 0.0] + 0.3 * u + 0.4 * w + rng.normal(size=n)
    d3 = Z @ [0.0, 0.1, 0.3, 0.7] - 0.4 * u + rng.normal(size=n)
    y = d1 - d2 + 0.5 * d3 + w + fe + u + rng.normal(size=25)[c]
    d = pd.DataFrame(
        dict(y=y, d1=d1, d2=d2, d3=d3, w=w, z1=Z[:, 0], z2=Z[:, 1], z3=Z[:, 2],
             z4=Z[:, 3], f=f, g=g, c=c, fc=f // 3, one=1)
    )  # fmt: skip
    d.to_csv(OUT, index=False, float_format="%.17g")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
