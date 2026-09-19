"""Write the simulated inputs of tests/reference_parity/test_did_synth_R_parity.py.

Run before ``../_generate_did_synth_R.R``. Deterministic (fixed seed); the CSV
is committed so the R and Python sides read identical bytes.

did_synth_sdid_panel.csv
    A balanced block-adoption panel with SEVERAL treated units (the Prop. 99
    replica used by Track A module 12 has one, which leaves synthdid's
    jackknife undefined and makes every placebo draw pick a single unit).
    Outcomes follow a two-factor interactive fixed-effects model, so unit
    weights, time weights and the intercepts all matter; the treated units
    load more heavily on the second factor, so plain DID is biased and the
    three synthdid estimators give visibly different answers.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent

rng = np.random.default_rng(20260918)
n_units, n_treated, n_periods, t0 = 30, 5, 14, 10
alpha = rng.normal(0.0, 1.0, n_units)
gamma = np.cumsum(rng.normal(0.2, 0.3, n_periods))
lam = rng.normal(0.0, 1.0, (n_units, 2))
lam[-n_treated:, 1] += 0.8
f = np.column_stack(
    [np.sin(np.arange(n_periods) / 2.0), np.linspace(-1.0, 1.5, n_periods)]
)
rows = []
for i in range(n_units):
    treated = i >= n_units - n_treated
    for t in range(n_periods):
        y = alpha[i] + gamma[t] + lam[i] @ f[t] + rng.normal(0.0, 0.3)
        d = int(treated and t >= t0)
        y += 1.5 * d
        rows.append({"unit": f"u{i + 1:02d}", "year": 2000 + t, "y": y, "treated": d})
pd.DataFrame(rows).to_csv(
    OUT / "did_synth_sdid_panel.csv", index=False, float_format="%.17g"
)

# did_synth_twfew_panel.csv — for the de Chaisemartin-D'Haultfoeuille weights.
# Groups are coarser than observations (several individuals per group x
# period cell, varying cell sizes), the panel is unbalanced (some cells are
# empty), observations carry sampling weights, and in two groups the
# treatment is only partly rolled out within the cell -- which is the case in
# which TwoWayFEWeights replaces D by its cell mean. Adoption is staggered and
# effects grow with exposure, so the TWFE coefficient carries negative
# weights.
rng = np.random.default_rng(20260919)
rows = []
n_groups, n_periods = 24, 8
adopt = rng.choice([3, 4, 5, 6, 0], size=n_groups, p=[0.2, 0.2, 0.2, 0.2, 0.2])
g_fe = rng.normal(0.0, 1.0, n_groups)
t_fe = np.cumsum(rng.normal(0.1, 0.2, n_periods))
for g in range(n_groups):
    for t in range(1, n_periods + 1):
        if rng.random() < 0.08:  # empty cell -> unbalanced
            continue
        n_cell = int(rng.integers(1, 5))
        for _ in range(n_cell):
            d = 1.0 if (adopt[g] > 0 and t >= adopt[g]) else 0.0
            if g in (3, 7) and d == 1.0 and rng.random() < 0.3:
                d = 0.0  # partial roll-out within the cell
            eff = d * (0.5 + 0.3 * (t - adopt[g])) if d else 0.0
            rows.append(
                {
                    "g": g + 1,
                    "t": t,
                    "y": g_fe[g] + t_fe[t - 1] + eff + rng.normal(0.0, 0.5),
                    "d": d,
                    "w": float(rng.uniform(0.5, 2.0)),
                }
            )
pd.DataFrame(rows).to_csv(
    OUT / "did_synth_twfew_panel.csv", index=False, float_format="%.17g"
)

# did_synth_honest_es.csv — a hand-crafted event study for sp.breakdown_m:
# three pre-period and three post-period coefficients (reference period
# omitted, HonestDiD's convention) with a dense covariance whose positive
# cross-period correlation is what separates the Rambachan-Roth FLCI from a
# per-period Wald interval. Written as betahat plus the covariance rows so R
# reads the identical doubles.
rel = np.array([-4, -3, -2, 0, 1, 2])
betahat = np.array([0.012, -0.018, 0.009, 0.21, 0.26, 0.31])
sd = np.array([0.020, 0.018, 0.016, 0.030, 0.034, 0.040])
rho = 0.45 ** np.abs(np.subtract.outer(np.arange(6), np.arange(6)))
sigma = rho * np.outer(sd, sd)
es = pd.DataFrame(sigma, columns=[f"s{j}" for j in range(6)])
es.insert(0, "betahat", betahat)
es.insert(0, "relative_time", rel)
es.to_csv(OUT / "did_synth_honest_es.csv", index=False, float_format="%.17g")
