"""Synthetic panels for the ``sp.panel_match`` parity fixtures.

Run this first, then ``_generate_panel_match_R.R`` (needs R with
``PanelMatch`` 3.1.5 in ``R_LIBS``), which builds the reference matched
sets, weights, estimates, analytic standard errors and balance tables and
writes ``panel_match_R.json``.

    python tests/reference_parity/_fixtures/_generate_panel_match_data.py
    R_LIBS=<lib with PanelMatch> Rscript \
        tests/reference_parity/_fixtures/_generate_panel_match_R.R

Treatment switches on and off (a two-state Markov chain), so a unit can be
treated several times; that is the design PanelMatch is built for.
``panel_match_gaps.csv`` is the same process with missing rows and missing
treatment / outcome / covariate values, for the missing-data rules.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def _panel(rng: np.random.Generator, n_units: int, n_periods: int) -> pd.DataFrame:
    rows = []
    for i in range(1, n_units + 1):
        a_i = rng.normal()
        d = int(rng.random() < 0.2)
        x1 = rng.normal()
        prev_d = d
        for t in range(1, n_periods + 1):
            x1 = 0.6 * x1 + rng.normal()
            x2 = rng.normal() + 0.3 * a_i
            # switching probabilities depend on the covariate
            p_on = 1 / (1 + np.exp(-(-1.8 + 0.6 * x1)))
            p_off = 0.25
            if t > 1:
                d = (
                    int(rng.random() < p_on)
                    if prev_d == 0
                    else int(rng.random() >= p_off)
                )
            y = (
                a_i
                + 0.15 * t
                + 0.5 * x1
                + 0.3 * x2
                + 1.0 * d
                + 0.4 * prev_d
                + rng.normal()
            )
            rows.append((i, t, d, round(y, 4), round(x1, 4), round(x2, 4), i % 3))
            prev_d = d
    return pd.DataFrame(rows, columns=["id", "t", "d", "y", "x1", "x2", "region"])


def main() -> None:
    rng = np.random.default_rng(20261019)
    full = _panel(rng, n_units=60, n_periods=16)
    full.to_csv(HERE / "panel_match_data.csv", index=False)

    gaps = _panel(rng, n_units=50, n_periods=14)
    # drop 4% of the rows outright (unbalanced panel) ...
    keep = rng.random(len(gaps)) > 0.04
    gaps = gaps[keep].copy()
    # ... and blank individual cells
    for col, share in (("y", 0.03), ("d", 0.02), ("x1", 0.03), ("x2", 0.02)):
        hit = rng.random(len(gaps)) < share
        gaps.loc[hit, col] = np.nan
    gaps.to_csv(HERE / "panel_match_gaps.csv", index=False)


if __name__ == "__main__":
    main()
