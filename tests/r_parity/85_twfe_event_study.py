"""StatsPAI dynamic TWFE event study (Python side) -- Module 85.

The most fundamental object in the DiD family had no pinned reference
value at all: not the coefficients, not their standard errors, not the
pre-treatment half. That matters more than its estimator status suggests.
Dynamic TWFE is the benchmark every other event study is *read against* --
Section 3 of the reference-convention work defines "TWFE-comparable" in
terms of it -- so an unpinned benchmark makes every comparison to it
unpinned too.

The reference is ``fixest::feols`` with an ``i(rel, ref = -1)``
interaction and two-way fixed effects, which is the specification
``sp.event_study`` documents and implements.

The window is chosen to cover every realised relative time, so no
outer-period binning happens on either side: ``sp.event_study`` clips the
outermost bins to the window, and a fixture that triggered that would be
comparing a bin average against a point coefficient.

Tolerance: rel < 1e-9 on estimate and SE. Both sides solve the same OLS
on the same rows with the same cluster-robust correction, so only
summation order should separate them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "85_twfe_event_study"
WINDOW = (-4, 4)


def make_data(n_units: int = 180, n_periods: int = 9, seed: int = PARITY_SEED):
    """Staggered panel with never-treated units and a deliberate pre-trend.

    The pre-trend is there so the lead coefficients are far from zero: a
    parity check where every pre-treatment cell is noise around zero
    cannot tell a correct lead from a badly attenuated one.
    """
    rng = np.random.default_rng(seed)
    # Non-staggered: half treated at period 5, half never treated.
    cohorts = np.where(np.arange(n_units) < n_units // 2, 5, 0)
    rows = []
    for i in range(n_units):
        g = int(cohorts[i])
        unit_fe = rng.normal(0.0, 1.0)
        for t in range(1, n_periods + 1):
            treated = 1 if (g > 0 and t >= g) else 0
            effect = 0.9 * (t - g + 1) if treated else 0.0
            pretrend = 0.20 * t if g > 0 else 0.0
            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "g": g,
                    "y": unit_fe + 0.3 * t + pretrend + effect + rng.normal(0.0, 0.6),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    # sp.event_study wants NaN for never-treated; the shared CSV uses 0.
    frame = df.copy()
    frame["g_nan"] = frame["g"].where(frame["g"] > 0, np.nan)

    fit = sp.event_study(
        frame, y="y", treat_time="g_nan", time="time", unit="unit",
        window=WINDOW, cluster="unit",
    )
    es = pd.DataFrame(fit.model_info["event_study"])
    es = es[~es["is_reference"]] if "is_reference" in es.columns else es

    rows: list[ParityRecord] = []
    for rel, att, se, n in zip(
        es["relative_time"], es["att"], es["se"], es.get("n_obs", [None] * len(es))
    ):
        k = int(rel)
        label = f"+{k}" if k >= 0 else str(k)
        rows.append(
            ParityRecord(
                module=MODULE, side="py", statistic=f"es_{label}",
                estimate=float(att), se=float(se),
                n=int(n) if n is not None and not pd.isna(n) else None,
            )
        )

    write_results(
        MODULE, "py", rows,
        extra={
            "reference": "fixest::feols(y ~ i(rel, ref=-1) | unit + time)",
            "window": list(WINDOW),
            "ref_period": -1,
            "cluster": "unit",
            "design": "non-staggered: one treated cohort plus never-treated",
            "window_note": (
                "The window covers every realised relative time (-4..+4), so "
                "every observation carries its own relative-time dummy and "
                "neither side has to decide what to do with out-of-window "
                "rows. sp.event_study reports single-point bins and leaves "
                "out-of-window rows in the reference category."
            ),
        },
    )


if __name__ == "__main__":
    main()
