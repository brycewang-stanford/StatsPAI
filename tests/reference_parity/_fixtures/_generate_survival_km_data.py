#!/usr/bin/env python
"""Deterministic survival fixture for KM / log-rank parity vs R ``survival``.

Two groups with different exponential hazards and independent exponential
censoring. Durations are rounded to 4 dp so the committed CSV is read
byte-identically by both ``sp.kaplan_meier`` / ``sp.logrank_test`` and the R
``survival::survfit`` / ``survival::survdiff`` reference generator. Rounding
also makes exact ties astronomically unlikely, so the product-limit estimator
and the Mantel-Haenszel log-rank statistic are both unambiguous on this design.

Run from the repository root:

    python tests/reference_parity/_fixtures/_generate_survival_km_data.py
"""

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "survival_km_data.csv"


def main() -> None:
    rng = np.random.default_rng(20260610)
    n_per = 120
    group = np.repeat([0, 1], n_per)
    # Group 1 has the higher hazard (shorter survival).
    scale = np.where(group == 0, 10.0, 6.0)
    latent = rng.exponential(scale)
    censor = rng.exponential(16.0, size=group.shape[0])  # per-obs censoring
    time = np.round(np.minimum(latent, censor), 4)
    status = (latent <= censor).astype(int)
    df = pd.DataFrame({"time": time, "status": status, "group": group})
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows, {df['status'].mean():.3f} event rate)")


if __name__ == "__main__":
    main()
