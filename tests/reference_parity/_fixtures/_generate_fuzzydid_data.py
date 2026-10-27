"""Synthetic fixtures for ``sp.fuzzy_did`` against Stata ``fuzzydid``.

Run from the repository root::

    python tests/reference_parity/_fixtures/_generate_fuzzydid_data.py
    stata-mp -b do tests/reference_parity/_fixtures/_generate_fuzzydid_Stata.do

Three designs, all repeated cross-sections with values rounded to three
decimals so the CSV bytes are the data both programs read:

* ``fuzzydid_2x2.csv`` -- two groups, two periods, binary treatment. The
  control group's treatment rate also moves (0.20 -> 0.35) and the treated
  and untreated trends differ, so Wald-DID, Wald-TC and Wald-CIC are three
  different numbers. Carries a continuous covariate ``x1``, a three-level
  covariate ``x2``, a cluster id ``cl`` and a binary outcome ``yb``.
* ``fuzzydid_ordered.csv`` -- two groups, two periods, treatment in
  {0, 1, 2, 3}, for the ordered-treatment Wald-TC / Wald-CIC and
  ``newcateg``.
* ``fuzzydid_multi.csv`` -- 40 groups over three periods, treatment in
  {0, 1, 2}; the group-level treatment distribution moves up, down or not
  at all between periods, coded by ``gt`` (change between t-1 and t, on the
  period-t rows) and ``gf`` (change between t and t+1, on the period-t
  rows), as in Section 4.2 of de Chaisemartin, D'Haultfoeuille and
  Guyonvarch (2019).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _two_by_two(rng: np.random.Generator) -> pd.DataFrame:
    sizes = {(0, 0): 300, (0, 1): 340, (1, 0): 280, (1, 1): 310}
    rates = {(0, 0): 0.20, (0, 1): 0.35, (1, 0): 0.25, (1, 1): 0.70}
    frames = []
    for (g, t), n in sizes.items():
        x1 = rng.normal(0.3 * g + 0.2 * t, 1.0, n)
        x2 = rng.integers(0, 3, n)
        p = np.clip(rates[(g, t)] + 0.08 * x1, 0.02, 0.98)
        d = rng.binomial(1, p)
        u = rng.normal(size=n)
        y = 1 + 0.5 * g + 0.8 * t + (1.5 + 0.7 * t) * d + 0.4 * t * d
        y = y + 0.6 * x1 + 0.3 * (x2 == 2) + u
        yb = (y + rng.logistic(size=n) > 2.5).astype(int)
        frames.append(
            pd.DataFrame({"y": y, "yb": yb, "g": g, "t": t, "d": d, "x1": x1, "x2": x2})
        )
    df = pd.concat(frames, ignore_index=True)
    df["cl"] = rng.integers(0, 150, len(df))
    return df


def _ordered(rng: np.random.Generator) -> pd.DataFrame:
    probs = {
        (0, 0): [0.40, 0.30, 0.20, 0.10],
        (0, 1): [0.35, 0.30, 0.20, 0.15],
        (1, 0): [0.45, 0.25, 0.20, 0.10],
        (1, 1): [0.15, 0.25, 0.30, 0.30],
    }
    frames = []
    for (g, t), p in probs.items():
        n = int(rng.integers(300, 420))
        x1 = rng.normal(size=n)
        d = rng.choice(4, size=n, p=p)
        y = 2 + 0.3 * g + 0.5 * t + d * (1.0 + 0.4 * t) + 0.5 * x1
        y = y + rng.normal(size=n) * (1 + 0.2 * d)
        frames.append(pd.DataFrame({"y": y, "g": g, "t": t, "d": d, "x1": x1}))
    return pd.concat(frames, ignore_index=True)


def _multi(rng: np.random.Generator) -> pd.DataFrame:
    frames = []
    for j in range(40):
        level = float(rng.uniform(0.3, 1.2))
        path = [level]
        for _ in range(2):
            u = rng.uniform()
            step = 0.0 if u < 0.5 else (0.6 if u < 0.75 else -0.6)
            path.append(float(np.clip(path[-1] + step, 0.1, 1.9)))
        for t in range(3):
            n = int(rng.integers(15, 31))
            m = path[t]
            # treatment in {0, 1, 2} with mean m (binomial(2, m / 2))
            d = rng.binomial(2, m / 2, n)
            x1 = rng.normal(0.1 * t, 1.0, n)
            y = 0.05 * j + 0.4 * t + d * (1.2 + 0.3 * t) + 0.5 * x1
            y = y + rng.normal(size=n)
            gt = np.nan if t == 0 else float(np.sign(path[t] - path[t - 1]))
            gf = np.nan if t == 2 else float(np.sign(path[t + 1] - path[t]))
            frames.append(
                pd.DataFrame(
                    {"y": y, "grp": j, "t": t, "d": d, "x1": x1, "gt": gt, "gf": gf}
                )
            )
    df = pd.concat(frames, ignore_index=True)
    for c in ("gt", "gf"):
        df[c] = df[c].astype("Int64")
    return df


def main() -> None:
    rng = np.random.default_rng(20261028)
    for name, df in (
        ("fuzzydid_2x2.csv", _two_by_two(rng)),
        ("fuzzydid_ordered.csv", _ordered(rng)),
        ("fuzzydid_multi.csv", _multi(rng)),
    ):
        for c in ("y", "x1"):
            df[c] = df[c].round(3)
        df.to_csv(HERE / name, index=False, float_format="%.3f")


if __name__ == "__main__":
    main()
