"""Balanced staggered panel for ``sp.bacon_decomposition(covariates=)``.

Run this, then ``_generate_bacon_controls_R.R`` (R ``bacondecomp`` 0.1.1,
MIT), which writes ``bacon_controls_R.json``. Forty units over twelve
periods; timing groups 1 (treated from the first period), 4, 7, 9 and
never treated; two time-varying covariates correlated with timing.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def main() -> None:
    rng = np.random.default_rng(20261019)
    cohorts = [1] * 5 + [4] * 9 + [7] * 8 + [9] * 8 + [0] * 10
    rows = []
    for i, g in enumerate(cohorts, start=1):
        a = rng.normal()
        for t in range(1, 13):
            d = int(g > 0 and t >= g)
            x1 = 0.3 * t + 0.5 * (g in (4, 7)) + rng.normal()
            x2 = rng.normal() + 0.2 * a
            y = (
                a
                + 0.1 * t
                + 0.4 * x1
                - 0.3 * x2
                + (1.0 + 0.1 * (t - g)) * d
                + rng.normal()
            )
            rows.append((i, t, d, round(y, 5), round(x1, 5), round(x2, 5)))
    pd.DataFrame(rows, columns=["id", "t", "d", "y", "x1", "x2"]).to_csv(
        HERE / "bacon_controls.csv", index=False
    )


if __name__ == "__main__":
    main()
