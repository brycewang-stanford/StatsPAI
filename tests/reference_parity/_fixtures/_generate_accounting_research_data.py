"""Synthetic data for test_accounting_research_parity.py.

Two files, both made up:

* ``accounting_research_panel.csv``: 150 firms over 12 years with a year
  effect and a persistent firm effect in the regressor and in the error
  (the design in which pooled, White, Fama-MacBeth and clustered standard
  errors disagree). ``gap`` marks 60 firm-years that the unbalanced tests
  drop.
* ``accounting_research_cross.csv``: 800 observations with 5% gross errors
  in ``y_out``, a count outcome ``fines`` that is zero whenever ``rare`` is
  1 (quasi-separation), an industry factor, a long-tailed variable with
  missing values, and a rare 0/1 event with a score.

Run from this folder: ``python _generate_accounting_research_data.py``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
rng = np.random.default_rng(20261006)

# ---- panel ---------------------------------------------------------------
N, T = 150, 12
firm = np.repeat(np.arange(1, N + 1), T)
year = np.tile(np.arange(2001, 2001 + T), N)
t_idx = year - 2001
year_x, year_e = rng.normal(0, 0.6, T), rng.normal(0, 1.0, T)


def ar1(rho: float, sd: float) -> np.ndarray:
    out = np.empty((N, T))
    out[:, 0] = rng.normal(0, sd, N)
    for t in range(1, T):
        out[:, t] = rho * out[:, t - 1] + rng.normal(0, sd * np.sqrt(1 - rho**2), N)
    return out.ravel()


x = year_x[t_idx] + ar1(0.6, 0.8)
z = rng.normal(size=N * T)
e = year_e[t_idx] + ar1(0.6, 1.2)
y = 0.5 + 1.0 * x - 0.3 * z + e
panel = pd.DataFrame({"firm": firm, "year": year, "y": y, "x": x, "z": z})
panel["gap"] = 0
panel.loc[rng.choice(len(panel), 60, replace=False), "gap"] = 1
panel.round(10).to_csv(HERE / "accounting_research_panel.csv", index=False)

# ---- cross-section -------------------------------------------------------
n = 800
x1, x2, x3 = rng.normal(size=(3, n))
ind = rng.integers(1, 6, n)
clean = 0.2 + 0.8 * x1 + 0.4 * x2 + 0.2 * x3 + 0.1 * (ind == 3) + rng.normal(size=n)
bad = np.zeros(n, dtype=int)
bad[rng.choice(n, 40, replace=False)] = 1
y_out = clean + bad * rng.normal(12, 2, n)
rare = (rng.uniform(size=n) < 0.06).astype(int)
mu = np.exp(0.3 + 0.5 * x1 - 0.2 * x2 + 0.1 * (ind == 2))
fines = rng.poisson(mu) * (1 - rare) * rng.integers(1, 40, n)
tail = np.exp(rng.normal(0, 1.5, n))
tail[rng.choice(n, 25, replace=False)] = np.nan
score = np.round(rng.uniform(size=n), 3)
event = (rng.uniform(size=n) < 1 / (1 + np.exp(4.5 - 4 * score))).astype(int)
cross = pd.DataFrame(
    {
        "y": clean,
        "y_out": y_out,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "ind": ind,
        "rare": rare,
        "fines": fines,
        "tail": tail,
        "score": score,
        "event": event,
    }
)
cross.round(10).to_csv(HERE / "accounting_research_cross.csv", index=False)
print(panel.shape, cross.shape, int(event.sum()), int(fines[rare == 1].sum()))
