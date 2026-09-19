"""Write the panels used by ``test_did_synth_didvar_parity.py``.

Run once from the repository root::

    python tests/reference_parity/_generate_did_synth_didvar_data.py

and then ``Rscript tests/reference_parity/_generate_did_synth_didvar_R.R``,
which reads these exact bytes back. The draws use a fixed seed; the CSVs are
committed, so the test never depends on this script being re-run.

Panels
------
``did_synth_didvar_contdose.csv``
    Balanced continuous-dose panel for ``sp.continuous_did``: 120 units x 6
    periods, 40% zero dose, time-invariant positive dose otherwise, a
    time-varying control ``x`` and a cluster ``region`` that nests the units.
``did_synth_didvar_contdose_unbal.csv``
    The same panel with a deterministic ~9% of rows removed (every unit keeps
    at least one pre and one post row). A one-pass ``y - ybar_i - ybar_t +
    ybar`` within transform is only exact on balanced panels; this panel is
    what tells the difference.
``did_synth_didvar_tvc.csv``
    Staggered panel for ``sp.did_timevarying_covariates``: 300 units x 7
    periods, cohorts {0 (never), 4, 5, 6}, two time-varying covariates whose
    level differs by cohort and which treatment moves after adoption, and an
    effect that is heterogeneous in the pre-treatment covariate. Under that
    heterogeneity a pooled ``dY ~ D + X`` regression and the Callaway-style
    outcome-regression ATT(g, t) are different numbers.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

_FIX = pathlib.Path(__file__).resolve().parent / "_fixtures"


def _contdose(seed: int = 20260918) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n, periods = 120, np.arange(1, 7)
    dose_u = np.where(rng.random(n) < 0.4, 0.0, rng.uniform(0.5, 5.0, n))
    alpha = rng.normal(0.0, 1.0, n)
    gamma = rng.normal(0.0, 0.5, periods.size)
    rows = []
    for i in range(n):
        for j, t in enumerate(periods):
            post = int(t >= 4)
            x = 0.3 * t + 0.2 * alpha[i] + rng.normal(0.0, 1.0)
            # Heterogeneous, mildly non-linear dose response plus noise.
            y = (
                alpha[i]
                + gamma[j]
                + 0.4 * x
                + post * (0.6 * dose_u[i] - 0.05 * dose_u[i] ** 2)
                + rng.normal(0.0, 1.0)
            )
            rows.append(
                {
                    "id": i + 1,
                    "time": int(t),
                    "y": y,
                    "dose": dose_u[i],
                    "x": x,
                    "region": i % 8 + 1,
                }
            )
    return pd.DataFrame(rows)


def _unbalance(df: pd.DataFrame) -> pd.DataFrame:
    drop = ((df["id"] * 7 + df["time"] * 3) % 11 == 0) & ~df["time"].isin([3, 4])
    out = df.loc[~drop].reset_index(drop=True)
    # Every unit must still carry a pre and a post row.
    chk = out.groupby("id")["time"].agg(lambda s: (s < 4).any() and (s >= 4).any())
    assert bool(chk.all())
    return out


def _tvc(seed: int = 20260919) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n, periods = 300, np.arange(1, 8)
    cohorts = rng.choice([0, 4, 5, 6], size=n, p=[0.35, 0.25, 0.2, 0.2])
    a = rng.normal(0.0, 1.0, n)
    lvl = {0: 0.0, 4: 1.0, 5: 0.5, 6: -0.5}
    rows = []
    for i in range(n):
        g = int(cohorts[i])
        x1_prev = None
        for t in periods:
            d = int(g > 0 and t >= g)
            # Covariates trend, differ in level by cohort, and are moved by
            # treatment once it starts (the "bad control" case).
            x1 = lvl[g] + 0.3 * t + 0.5 * a[i] + rng.normal(0.0, 0.7) + 1.5 * d
            x2 = 0.2 * t * (1 + lvl[g]) + rng.normal(0.0, 1.0) - 0.8 * d
            if t == g - 1:
                x1_prev = x1
            base = x1_prev if (d and x1_prev is not None) else x1
            effect = d * (1.0 + 0.5 * base + 0.2 * (t - g))
            y = a[i] + 0.25 * t + 0.4 * x1 * (1 - d) + 0.3 * x2 + effect
            y += 0.15 * t * lvl[g] + rng.normal(0.0, 1.0)
            rows.append(
                {"id": i + 1, "time": int(t), "g": g, "y": y, "x1": x1, "x2": x2}
            )
    return pd.DataFrame(rows)


def main() -> None:
    cont = _contdose()
    cont.to_csv(
        _FIX / "did_synth_didvar_contdose.csv", index=False, float_format="%.17g"
    )
    _unbalance(cont).to_csv(
        _FIX / "did_synth_didvar_contdose_unbal.csv", index=False, float_format="%.17g"
    )
    _tvc().to_csv(_FIX / "did_synth_didvar_tvc.csv", index=False, float_format="%.17g")
    print("wrote did_synth_didvar_{contdose,contdose_unbal,tvc}.csv")


if __name__ == "__main__":
    main()
