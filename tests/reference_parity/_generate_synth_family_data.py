"""Write the simulated inputs of test_synth_family_R_parity.py.

Run before ``_generate_synth_family_R.R``. Deterministic (fixed seeds); the
CSVs are committed so the R and Python sides read identical bytes.

synth_family_asc.csv
    18 units, 20 periods, treatment from period 14 for units 1 and 2
    (``treated == 1`` from then on). Two-factor model with unit effects;
    units 1-2 load on the factors as convex combinations of donors plus a
    perturbation, so neither SCM nor ridge fits exactly. Effect 2 + 0.1
    (t - 14). ``x1`` is a time-varying covariate correlated with the
    loadings, ``x2`` a time-invariant one. Used by ``sp.augsynth`` with
    several treated units, ``progfunc``, ``scm``, ``fixedeff`` and
    ``covariates`` (vs ``augsynth::augsynth``).

synth_family_fect.csv
    24 units, 14 periods. Units 1-8 never treated; units 9-16 adopt for
    good at periods 6-9 (two units each); units 17-24 are treated for three
    or four periods starting at 5-8 and then switch off (reversals), with
    an effect that lingers one period after switch-off (0.8). Two-way fixed
    effects plus noise, effect 2 while treated. Used by ``sp.fect``'s
    placebo, carryover and leave-one-period-out tests (vs ``fect::fect``
    with ``vartype = "jackknife"``, which is deterministic).
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "_fixtures"


def asc_panel() -> pd.DataFrame:
    rng = np.random.default_rng(20261019)
    n, T, t0 = 18, 20, 14
    f = np.cumsum(rng.normal(0, 1, (T, 2)), axis=0)
    lam = rng.normal(0, 1, (n, 2))
    alpha = rng.normal(5, 2, n)
    for i in (0, 1):
        w = rng.dirichlet(np.ones(4))
        donors = rng.choice(np.arange(2, n), size=4, replace=False)
        lam[i] = w @ lam[donors] + rng.normal(0, 0.3, 2)
    rows = []
    for i in range(n):
        x2 = float(lam[i, 0] + rng.normal(0, 0.5))
        for t in range(T):
            treated = int(i < 2 and t + 1 >= t0)
            y = alpha[i] + f[t] @ lam[i] + rng.normal(0, 0.4)
            y += treated * (2.0 + 0.1 * (t + 1 - t0))
            x1 = 0.5 * lam[i, 1] + 0.1 * t + rng.normal(0, 0.3)
            rows.append(
                (i + 1, t + 1, round(y, 10), treated, round(x1, 10), round(x2, 10))
            )
    return pd.DataFrame(rows, columns=["unit", "time", "y", "treated", "x1", "x2"])


def fect_panel() -> pd.DataFrame:
    rng = np.random.default_rng(1019)
    n, T = 24, 14
    alpha = rng.normal(0, 1, n)
    xi = np.cumsum(rng.normal(0.2, 0.5, T))
    D = np.zeros((n, T), dtype=int)
    for k, i in enumerate(range(8, 16)):
        D[i, 5 + k // 2 :] = 1  # adopt at periods 6, 7, 8, 9 (1-based)
    for k, i in enumerate(range(16, 24)):
        start = 4 + k // 2
        D[i, start : start + 3 + (k % 2)] = 1
    rows = []
    for i in range(n):
        for t in range(T):
            carry = 0.8 if (t > 0 and D[i, t - 1] == 1 and D[i, t] == 0) else 0.0
            y = alpha[i] + xi[t] + 2.0 * D[i, t] + carry + rng.normal(0, 0.5)
            rows.append((i + 1, t + 1, round(float(y), 10), int(D[i, t])))
    return pd.DataFrame(rows, columns=["id", "time", "Y", "D"])


if __name__ == "__main__":
    asc_panel().to_csv(OUT / "synth_family_asc.csv", index=False)
    fect_panel().to_csv(OUT / "synth_family_fect.csv", index=False)
