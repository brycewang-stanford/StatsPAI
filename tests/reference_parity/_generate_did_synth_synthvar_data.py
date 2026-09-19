"""Write the simulated inputs of test_did_synth_synthvar_parity.py.

Run before ``_generate_did_synth_synthvar_R.R``. Deterministic (fixed seed);
the CSVs are committed so the R and Python sides read identical bytes.

did_synth_synthvar_single.csv
    One treated unit (``unit == 1``) and eight donors, 30 periods, treatment
    from period 21 (20 pre-periods). Two-factor interactive model plus a
    unit level; the treated unit's level sits far above every donor, so the
    level-matching SCM and the de-meaned SCM give different answers and the
    de-meaned problem has a unique interior-ish solution (T0 = 20 > J = 8).
    Used by ``demeaned_synth`` (vs augsynth fixedeff=TRUE, progfunc="None")
    and ``robust_synth`` (vs scpi OLS-with-constant and glmnet).

did_synth_synthvar_stag.csv
    Staggered adoption: 24 units, 22 periods; units 1-3 adopt at period 9,
    units 4-6 at period 13, units 7-24 never treated. Used by
    ``staggered_synth`` (vs augsynth::multisynth).

did_synth_synthvar_micro.csv
    Individual-level repeated cross-sections for Gunsilius' distributional
    synthetic control: unit 1 treated, units 2-6 controls, periods 1-6,
    t0 = 4 (3 pre-periods), 250 draws per unit-period. Control outcome
    distributions differ in location, scale and skew; the treated unit is a
    quantile mixture of controls 2-4 before treatment and gets a
    distribution-changing effect afterwards. Used by ``discos`` (vs
    DiSCos::DiSCo).
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "_fixtures"


def _single() -> pd.DataFrame:
    rng = np.random.default_rng(20260918)
    n_units, n_periods, t0 = 9, 30, 20
    f = np.column_stack(
        [
            np.cumsum(rng.normal(0.3, 1.0, n_periods)),
            np.sin(np.arange(n_periods) / 3.0) * 3.0,
        ]
    )
    load = rng.uniform(0.2, 1.8, (n_units, 2))
    level = rng.uniform(0.0, 10.0, n_units)
    level[0] = 25.0
    rows = []
    for i in range(n_units):
        for t in range(n_periods):
            y = level[i] + load[i] @ f[t] + rng.normal(0.0, 0.5)
            d = int(i == 0 and t >= t0)
            y += d * (2.0 + 0.1 * (t - t0))
            rows.append({"unit": i + 1, "time": t + 1, "y": y, "treated": d})
    return pd.DataFrame(rows)


def _stag() -> pd.DataFrame:
    rng = np.random.default_rng(20260919)
    n_units, n_periods = 24, 22
    adopt = {1: 9, 2: 9, 3: 9, 4: 13, 5: 13, 6: 13}
    f = np.column_stack(
        [
            np.cumsum(rng.normal(0.2, 1.0, n_periods)),
            np.cos(np.arange(n_periods) / 2.5) * 2.0,
        ]
    )
    load = rng.uniform(0.3, 1.7, (n_units, 2))
    level = rng.normal(5.0, 2.0, n_units)
    rows = []
    for i in range(n_units):
        uid = i + 1
        g = adopt.get(uid)
        for t in range(n_periods):
            per = t + 1
            y = level[i] + load[i] @ f[t] + rng.normal(0.0, 0.4)
            d = int(g is not None and per >= g)
            if d:
                y += 1.0 + 0.25 * (per - g)
            rows.append({"unit": uid, "time": per, "y": y, "treated": d})
    return pd.DataFrame(rows)


def _micro() -> pd.DataFrame:
    rng = np.random.default_rng(20260920)
    n_obs, periods, t0 = 250, range(1, 7), 4
    # control j: location mu_j + 0.4 t, scale s_j, skew via a lognormal part
    mu = {2: 0.0, 3: 2.0, 4: 4.5, 5: 1.0, 6: 3.0}
    sd = {2: 1.0, 3: 0.6, 4: 1.5, 5: 2.0, 6: 0.8}
    skew = {2: 0.0, 3: 0.8, 4: 0.3, 5: 0.0, 6: 1.2}

    def draw(j: int, t: int, n: int) -> np.ndarray:
        base = rng.normal(mu[j] + 0.4 * t, sd[j], n)
        return base + skew[j] * rng.lognormal(0.0, 0.5, n)

    rows = []
    for t in periods:
        for j in mu:
            for v in draw(j, t, n_obs):
                rows.append({"id": j, "time": t, "y": v})
        # treated: quantile mixture 0.5 Q2 + 0.3 Q3 + 0.2 Q4 of fresh draws
        u = np.sort(rng.uniform(size=n_obs))
        q = (
            0.5 * np.quantile(draw(2, t, 4000), u)
            + 0.3 * np.quantile(draw(3, t, 4000), u)
            + 0.2 * np.quantile(draw(4, t, 4000), u)
        )
        if t >= t0:
            q = q + 0.5 + 0.8 * u  # upper quantiles move more
        for v in rng.permutation(q):
            rows.append({"id": 1, "time": t, "y": v})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    _single().to_csv(
        OUT / "did_synth_synthvar_single.csv", index=False, float_format="%.17g"
    )
    _stag().to_csv(
        OUT / "did_synth_synthvar_stag.csv", index=False, float_format="%.17g"
    )
    _micro().to_csv(
        OUT / "did_synth_synthvar_micro.csv", index=False, float_format="%.17g"
    )
