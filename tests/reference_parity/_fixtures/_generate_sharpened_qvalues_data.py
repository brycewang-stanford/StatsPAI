"""Input p-vectors for the sharpened q-value parity test.

Writes ``sharpened_qvalues_input.csv`` (columns ``case, pval``; an empty
``pval`` is a missing p-value). Each value is written as the shortest
string that round-trips the double, which Stata reads back exactly under
``set type double``. Run from the repository root::

    python tests/reference_parity/_fixtures/_generate_sharpened_qvalues_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np

OUT = pathlib.Path(__file__).parent / "sharpened_qvalues_input.csv"


def cases() -> dict:
    rng = np.random.default_rng(20261028)
    mixed = np.concatenate([rng.beta(0.3, 8.0, 25), rng.uniform(size=15)])
    big = np.concatenate([rng.beta(0.4, 10.0, 120), rng.uniform(size=180)])
    huge = np.concatenate([rng.beta(0.5, 20.0, 400), rng.uniform(size=1600)])
    with_nan = [0.01, np.nan, 0.04, 0.02, np.nan, 0.3, 0.004]
    return {
        "five": [0.001, 0.01, 0.03, 0.04, 0.2],
        "mixed40": mixed.tolist(),
        "ties": [0.01, 0.01, 0.02, 0.02, 0.02, 0.5, 0.5],
        "zero_one": [0.0, 0.0, 0.003, 0.2, 1.0, 1.0],
        "single_small": [0.03],
        "single_large": [0.7],
        "single_zero": [0.0],
        "all_tiny": [1e-6, 2e-5, 1e-4, 0.0005],
        "all_large": [0.6, 0.7, 0.8, 0.9],
        # p-values that sit exactly on thresholds q/(1+q) r/M for round q
        "boundary": [0.0125, 0.025, 0.0375, 0.05, 0.25, 0.5],
        "anderson_like": [
            0.001,
            0.008,
            0.039,
            0.041,
            0.042,
            0.06,
            0.074,
            0.205,
            0.212,
            0.216,
            0.222,
            0.251,
            0.269,
            0.275,
            0.34,
            0.341,
            0.384,
            0.569,
            0.594,
            0.696,
            0.762,
            0.94,
            0.942,
            0.975,
            0.986,
        ],
        "many_small": (np.arange(1, 61) / 6000.0).tolist(),
        "big300": big.tolist(),
        "huge2000": huge.tolist(),
        "with_nan": with_nan,
        # the same vector with each missing p-value replaced by 1: what
        # missing='count' computes before putting the NaN back
        "with_nan_as_one": [1.0 if np.isnan(v) else v for v in with_nan],
    }


def main() -> None:
    lines = ["case,pval"]
    for name, vals in cases().items():
        for v in vals:
            lines.append(f"{name}," + ("" if np.isnan(v) else repr(float(v))))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
