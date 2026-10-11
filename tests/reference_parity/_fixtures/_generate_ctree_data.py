"""Synthetic data for the ``sp.ctree`` parity fixtures.

Run this first, then ``_generate_ctree.R`` (needs R with ``partykit``),
which fits the reference trees and writes ``ctree_R.json``.

    python tests/reference_parity/_fixtures/_generate_ctree_data.py
    Rscript tests/reference_parity/_fixtures/_generate_ctree.R
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def main() -> None:
    rng = np.random.default_rng(20261012)

    # Numeric outcome; ties, a four-level unordered covariate, a binary one.
    n = 300
    reg = pd.DataFrame(
        {
            "x1": rng.normal(size=n).round(3),
            "x2": rng.uniform(size=n).round(3),
            "x3": rng.integers(0, 6, size=n),
            "x4": rng.normal(size=n).round(3),
            "f": rng.choice(["a", "b", "c", "d"], size=n),
            "b": rng.integers(0, 2, size=n),
        }
    )
    reg["y"] = (
        1.5 * (reg["x1"] > 0.2)
        + 1.0 * reg["f"].isin(["b", "d"])
        + 0.8 * (reg["x3"] >= 4) * reg["b"]
        + rng.normal(size=n)
    ).round(4)
    reg.to_csv(HERE / "ctree_reg.csv", index=False)

    # Three-class outcome; a five-level unordered and an ordered covariate.
    n = 400
    multi = pd.DataFrame(
        {
            "x1": rng.normal(size=n).round(3),
            "x2": rng.normal(size=n).round(3),
            "g": rng.choice(["p", "q", "r", "s", "t"], size=n),
            "grade": rng.choice(["low", "mid", "high", "top"], size=n),
            "x5": rng.integers(0, 10, size=n),
        }
    )
    score = (
        1.2 * multi["x1"]
        + 1.0 * multi["g"].isin(["q", "t"])
        + 0.9 * multi["grade"].isin(["high", "top"])
        + rng.normal(size=n)
    )
    multi["y"] = np.where(score < 0.2, "u", np.where(score < 1.6, "v", "w"))
    multi.to_csv(HERE / "ctree_multi.csv", index=False)

    # Small binary-outcome sample, where the limits on node size bind.
    n = 150
    small = pd.DataFrame(
        {
            "x1": rng.normal(size=n).round(2),
            "x2": rng.integers(0, 4, size=n),
            "x3": rng.uniform(size=n).round(2),
            "rare": (rng.uniform(size=n) < 0.06).astype(int),
        }
    )
    p = 1 / (1 + np.exp(-(1.4 * small["x1"] + 2.5 * small["rare"] - 0.6 * small["x2"])))
    small["y"] = (rng.uniform(size=n) < p).astype(int)
    small.to_csv(HERE / "ctree_small.csv", index=False)


if __name__ == "__main__":
    main()
