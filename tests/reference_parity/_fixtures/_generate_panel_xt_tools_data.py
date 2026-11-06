"""Panel behind ``test_panel_xt_tools_stata_parity.py`` (2026-11-05).

``python _generate_panel_xt_tools_data.py`` (from this directory) writes
``panel_xt_tools.csv``, read by both the Stata generator
``_generate_panel_xt_tools_stata.do`` and the test, so the two sides fit
the same bytes.

120 panels. Panels 1-60 are balanced (six periods, no gaps, ``bal = 1``);
panels 61-120 have 3 to 9 periods with interior gaps. The disturbance is
AR(1) with coefficient 0.4. ``x1`` and ``w`` are time-varying and
exogenous, ``x2`` time-varying and correlated with the panel effect,
``ti`` time-invariant and exogenous, ``zi`` time-invariant and correlated
with the panel effect; ``xm`` is ``x1`` with about a tenth of its values
missing; ``g`` groups four consecutive panels (30 clusters).
"""

from pathlib import Path

import numpy as np
import pandas as pd


def make_panel(seed: int = 20261105) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(120):
        mu = rng.normal()
        ti = rng.normal()
        zi = 0.6 * mu + rng.normal()
        if i < 60:
            periods = np.arange(1, 7)
        else:
            T = int(rng.integers(3, 10))
            keep = rng.uniform(size=T) > 0.2
            keep[0] = True
            periods = np.arange(1, T + 1)[keep]
            if len(periods) < 2:
                periods = np.arange(1, 3)
        e = rng.normal() / np.sqrt(1 - 0.16)
        last = 1
        for t in periods:
            for _ in range(int(t - last)):
                e = 0.4 * e + rng.normal()
            last = t
            x1 = rng.normal()
            w = rng.normal()
            x2 = 0.5 * mu + rng.normal()
            y = 1.0 + 0.8 * x1 - 0.5 * w + 0.6 * x2 + 0.4 * ti - 0.7 * zi + mu + e
            xm = np.nan if rng.uniform() < 0.1 else x1
            rows.append((i + 1, int(t), y, x1, w, x2, ti, zi, xm, i // 4 + 1))
    cols = ["id", "t", "y", "x1", "w", "x2", "ti", "zi", "xm", "g"]
    df = pd.DataFrame(rows, columns=cols)
    df["bal"] = (df["id"] <= 60).astype(int)
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "panel_xt_tools.csv"
    make_panel().to_csv(out, index=False)
