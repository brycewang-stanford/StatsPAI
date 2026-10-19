"""Data for the ``sp.mediate_paths`` fixture (R ``paths``).

800 units: confounded binary treatment ``a`` (covariates ``x1``, ``x2``),
mediator ``m1`` and a second block ``m1b``, a later mediator ``m2`` that
depends on ``m1``, a continuous outcome ``y`` and a binary one ``yb``,
both with treatment x mediator interactions.

    python tests/reference_parity/_fixtures/_generate_mediate_paths_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "mediate_paths.csv"


def main() -> None:
    rng = np.random.default_rng(11)
    n = 800
    x1 = rng.normal(size=n)
    x2 = rng.integers(0, 2, n)
    a = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * x1 - 0.3 * x2))), n)
    m1 = 0.6 * a + 0.4 * x1 + rng.normal(size=n)
    m1b = 0.3 * a - 0.2 * x2 + rng.normal(size=n)
    m2 = 0.5 * a + 0.7 * m1 + 0.2 * x2 + 0.3 * a * m1 + rng.normal(size=n)
    y = (
        0.4 * a + 0.5 * m1 + 0.8 * m2 + 0.3 * a * m2 + 0.2 * m1b
        + 0.3 * x1 + 0.2 * x2 + rng.normal(size=n)
    )
    eta = -0.5 + 0.4 * a + 0.3 * m1 + 0.4 * m2 + 0.2 * a * m2 + 0.2 * x1
    yb = rng.binomial(1, 1 / (1 + np.exp(-eta)), n)
    pd.DataFrame(
        {"y": y, "yb": yb, "a": a, "m1": m1, "m1b": m1b, "m2": m2, "x1": x1, "x2": x2}
    ).to_csv(OUT, index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
