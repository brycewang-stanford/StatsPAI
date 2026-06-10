#!/usr/bin/env python
"""Deterministic Weibull-AFT fixture for parametric-survival parity vs R.

A Weibull accelerated-failure-time data-generating process with one
continuous and one binary covariate, plus independent exponential
censoring (~25%). Times are rounded to 4 dp so the committed CSV is read
byte-identically by ``sp.survreg`` and R ``survival::survreg``. Parity (not
recovery) is the target: both fit the same AFT log-likelihood on these
exact bytes.

Run from the repository root:

    python tests/reference_parity/_fixtures/_generate_survreg_aft_data.py
"""

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "survreg_aft_data.csv"


def main() -> None:
    rng = np.random.default_rng(20260610)
    n = 400
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.integers(0, 2, n).astype(float)
    # AFT linear predictor on the log-time scale; Weibull shape k = 1/sigma.
    lp = 2.0 + 0.5 * x1 - 0.4 * x2
    sigma = 0.8
    k = 1.0 / sigma
    scale = np.exp(lp)
    latent = scale * rng.weibull(k, size=n)
    censor = rng.exponential(np.exp(2.3), size=n)  # ~25% censoring
    time = np.round(np.minimum(latent, censor), 4)
    status = (latent <= censor).astype(int)
    df = pd.DataFrame({"time": time, "status": status, "x1": np.round(x1, 4), "x2": x2})
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows, {df['status'].mean():.3f} event rate)")


if __name__ == "__main__":
    main()
