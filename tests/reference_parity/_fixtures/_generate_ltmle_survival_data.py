"""Data for ``test_ltmle_survival_R_parity.py``.

1,200 subjects, three intervals. Per interval: a covariate ``L_k`` that
depends on past treatment, a treatment ``A_k`` that depends on ``L_k`` and
on the last treatment, an indicator ``C_k`` (1 = still observed) and the
cumulative event indicator ``Y_k`` (1 from the event onwards, as R
``ltmle`` wants it). After the event or after censoring, the later
covariates, treatments and censoring indicators are empty; ``Y`` stays 1
after the event and is empty after censoring. Values have six decimals.

    python tests/reference_parity/_fixtures/_generate_ltmle_survival_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":
    rng = np.random.default_rng(20261114)
    n, K = 1200, 3
    expit = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    W = np.round(rng.normal(size=n), 6)
    cols = {"W": W}
    alive = np.ones(n, dtype=bool)  # no event so far
    seen = np.ones(n, dtype=bool)  # not censored so far
    a_prev = np.zeros(n)
    for k in range(1, K + 1):
        active = alive & seen
        L = np.round(0.4 * W - 0.5 * a_prev + rng.normal(size=n), 6)
        A = (rng.uniform(size=n) < expit(-0.2 + 0.5 * L + 0.6 * a_prev)).astype(float)
        C = (rng.uniform(size=n) < expit(2.2 + 0.3 * L - 0.2 * A)).astype(float)
        hazard = expit(-1.6 + 0.5 * W + 0.4 * L - 0.7 * A - 0.3 * a_prev)
        event = rng.uniform(size=n) < hazard
        cols[f"L{k}"] = np.where(active, L, np.nan)
        cols[f"A{k}"] = np.where(active, A, np.nan)
        cols[f"C{k}"] = np.where(active, C, np.nan)
        seen_now = seen & ((C == 1) | ~alive)
        # cumulative event indicator: 1 after the event, empty once censored
        y = np.where(~alive, 1.0, np.where(seen_now, event.astype(float), np.nan))
        cols[f"Y{k}"] = y
        alive = alive & ~(seen_now & event)
        seen = seen_now
        a_prev = A
    df = pd.DataFrame(cols)
    df.to_csv(HERE / "ltmle_survival.csv", index=False)
    print(df.notna().sum().to_dict())
    print(df[["Y1", "Y2", "Y3"]].mean().to_dict())
