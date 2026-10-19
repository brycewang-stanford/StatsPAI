"""Data for the nonlinear four-way decomposition fixture (Stata med4way).

One sample of 1,500: binary exposure ``a`` (confounded by ``c1``), a normal
mediator ``mc`` and a binary one ``mb``, and for each mediator a binary
outcome (rare-ish logistic: about 20% events) and a count outcome, all with
an exposure-mediator interaction; ``yc`` is a linear outcome.

    python tests/reference_parity/_fixtures/_generate_four_way_nonlinear_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "four_way_nonlinear.csv"


def main() -> None:
    rng = np.random.default_rng(20261019)
    n = 1500
    c1 = rng.normal(size=n)
    c2 = rng.integers(0, 2, n)
    a = rng.binomial(1, 1 / (1 + np.exp(-(0.3 * c1 - 0.2))), n)
    mc = 0.5 + 0.8 * a + 0.3 * c1 - 0.2 * c2 + rng.normal(size=n)
    mb = rng.binomial(1, 1 / (1 + np.exp(-(-0.4 + 0.9 * a + 0.3 * c1 - 0.3 * c2))), n)

    def binary(m):
        eta = -2.5 + 0.4 * a + 0.5 * m + 0.3 * a * m + 0.2 * c1 + 0.1 * c2
        return rng.binomial(1, 1 / (1 + np.exp(-eta)), n)

    def count(m):
        return rng.poisson(
            np.exp(-0.5 + 0.3 * a + 0.25 * m + 0.15 * a * m + 0.2 * c1 - 0.1 * c2)
        )

    yc = 1 + 0.5 * a + 0.7 * mc + 0.4 * a * mc + 0.3 * c1 + rng.normal(size=n)
    pd.DataFrame(
        {
            "a": a,
            "mc": mc,
            "mb": mb,
            "c1": c1,
            "c2": c2,
            "yc": yc,
            "ybc": binary(mc),
            "ybb": binary(mb),
            "ypc": count(mc),
            "ypb": count(mb),
        }
    ).to_csv(OUT, index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
