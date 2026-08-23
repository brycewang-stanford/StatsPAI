"""StatsPAI BJS pre-trend parity (Python side) -- Module 84.

Module 16 pins the BJS *pooled ATT*. This one pins the object that
defect actually lived in: the pre-treatment lead vector.

``sp.did_imputation`` reports two things a reader acts on separately. The
lags are imputation residuals. The leads are a different construction
entirely, and until v1.23.0 StatsPAI built them the ``fect``/``did2s``
way -- means of the *in-sample* residual -- while documenting Stata
``did_imputation, pretrends(k)``. Those two disagree by exactly the
untreated unit share (Li & Strezhnev 2025; Roth 2026, appendix A), which
attenuates reported pre-trends toward zero. Nothing in the archive could
see it, because the archive pinned a scalar and the defect was in a
vector.

Both reference implementations are pinned here, because the two answer
the same question through different machinery: Stata ``did_imputation``
(Borusyak, SSC) and R ``didimputation::did_imputation`` (Butts &
Borusyak, CRAN).

Fixture design. Cohorts 6 and 8 plus never-treated, twelve periods, so
every treated cohort has strictly more pre-treatment periods than the
three requested leads. That matters: leads covering a cohort's whole
pre-history are collinear with its unit effects, and Stata's
``autosample`` would silently drop rows to cope. Here it has nothing to
do, so all three sides estimate on identical rows.

Tolerance: rel < 1e-6 for both estimate and SE. The SE agreement is
not incidental -- building this module is what surfaced the variance
approximation that v1.23.0 replaced with the exact BJS weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "84_bjs_pretrends"
LEADS = (-3, -2, -1)
HORIZONS = (0, 1, 2, 3)


def make_data(n_units: int = 200, n_periods: int = 12, seed: int = PARITY_SEED) -> pd.DataFrame:
    """Staggered panel with a deliberate pre-trend to make the leads bite.

    The treated cohorts drift relative to the never-treated group before
    treatment, so the lead coefficients are far from zero and a
    construction that attenuates them toward zero is visible rather than
    lost in noise.
    """
    rng = np.random.default_rng(seed)
    cohorts = rng.choice([6, 8, 0], size=n_units, p=[1 / 3, 1 / 3, 1 / 3])
    rows = []
    for i in range(n_units):
        g = int(cohorts[i])
        unit_fe = rng.normal(0.0, 1.0)
        for t in range(1, n_periods + 1):
            treated = 1 if (g > 0 and t >= g) else 0
            effect = 1.2 + 0.4 * (t - g) if treated else 0.0
            pretrend = 0.25 * t if g > 0 else 0.0
            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "g": g,
                    "y": unit_fe + 0.3 * t + pretrend + effect + rng.normal(0.0, 0.5),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    fit = sp.did_imputation(
        df,
        y="y",
        group="unit",
        time="time",
        first_treat="g",
        horizon=[*LEADS, *HORIZONS],
        cluster="unit",
        pretrend_method="bjs",
    )
    es = pd.DataFrame(fit.model_info["event_study"]).set_index("relative_time")

    rows: list[ParityRecord] = []
    for k in (*LEADS, *HORIZONS):
        if k not in es.index:
            raise SystemExit(f"did_imputation returned no relative time {k}")
        row = es.loc[k]
        label = f"pre{abs(int(k))}" if k < 0 else f"tau{int(k)}"
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"{label}_att",
                estimate=float(row["att"]),
                se=float(row["se"]),
                n=int(row["n_obs"]),
            )
        )

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "pretrend_method": "bjs",
            "cluster": "unit",
            "leads": list(LEADS),
            "horizons": list(HORIZONS),
            "statspai_version": getattr(sp, "__version__", None),
            "note": (
                "Pins the pre-treatment lead vector, not just the pooled ATT "
                "of module 16. pre<k>_att is relative time -k."
            ),
        },
    )


if __name__ == "__main__":
    main()
