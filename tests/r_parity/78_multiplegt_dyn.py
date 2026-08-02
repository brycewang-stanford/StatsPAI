"""StatsPAI dCDH intertemporal event-study parity (Python side) -- Module 78.

Runs ``sp.did_multiplegt_dyn`` on a staggered binary-treatment panel; the
companion R script runs the authors' own ``DIDmultiplegtDYN::did_multiplegt_dyn``
on the same dumped CSV.

Index convention: the R package labels effects from 1, so its ``Effect_k``
is StatsPAI's horizon ``k - 1``, and its ``Placebo_k`` is horizon ``-k``.
The statistic names here use the R labelling so the two sides line up.

The headline aggregate is emitted under ``aggregation="switchers"``, which
is the package's ``Av_tot_eff``; StatsPAI's default equal-weight average is
emitted alongside so the harness records the convention gap rather than
hiding it.

Standard errors are not compared: ``DIDmultiplegtDYN`` reports analytical
influence-function SEs and ``sp.did_multiplegt_dyn`` has only a cluster
bootstrap.

Tolerance: rel < 1e-6 on every effect, placebo and the switcher-weighted
aggregate.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from _common import ParityRecord, dump_csv, write_results

MODULE = "78_multiplegt_dyn"
N_EFFECTS = 4
N_PLACEBOS = 2
# Second design: treatment switches OFF as well as on. Without it the
# switch-off branch -- the baseline-matched control group and the
# divide-by-delta-D sign convention -- goes untested, and that branch is
# exactly what separates this estimator from every cohort-based one.
OFF_EFFECTS = 2
OFF_PLACEBOS = 1


def _panel() -> pd.DataFrame:
    """Staggered absorbing treatment, cohorts {3, 5, 7} plus never-treated.

    Later cohorts run out of horizon before earlier ones, so the number of
    switchers behind each horizon falls with l -- which is exactly what
    makes the two aggregation conventions disagree.
    """
    rng = np.random.default_rng(4)
    rows = []
    for i in range(1, 201):
        g = int(rng.choice([3, 5, 7, 0]))
        fe = rng.normal()
        for t in range(1, 9):
            d = int(g > 0 and t >= g)
            y = fe + 0.2 * t + 1.5 * d + rng.normal(0, 0.7)
            rows.append({"id": i, "t": t, "d": d, "y": y})
    return pd.DataFrame(rows)


def _switching_panel() -> pd.DataFrame:
    """Treatment turns on and off. Same DGP as module 81's fixture."""
    rng = np.random.default_rng(9)
    rows = []
    for uid in range(1, 181):
        fe = rng.normal()
        d_prev = 0
        for t in range(1, 7):
            if t == 1:
                d = int(rng.random() < 0.3)
            elif rng.random() < 0.25:
                d = 1 - d_prev
            else:
                d = d_prev
            d_prev = d
            rows.append(
                {
                    "id": uid,
                    "t": t,
                    "d": d,
                    "y": fe + 0.25 * t + 1.2 * d + rng.normal(0, 0.6),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    import statspai as sp

    df = _panel()
    dump_csv(df, MODULE)

    fits = {}
    for agg in ("switchers", "simple"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits[agg] = sp.did_multiplegt_dyn(
                df,
                y="y",
                group="id",
                time="t",
                treatment="d",
                dynamic=N_EFFECTS - 1,
                placebo=N_PLACEBOS,
                cluster="id",
                n_boot=0,
                aggregation=agg,
            )

    detail = fits["switchers"].detail.set_index("horizon")
    rows: list[ParityRecord] = []
    for k in range(1, N_EFFECTS + 1):
        row = detail.loc[k - 1]
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"Effect_{k}",
                estimate=float(row["delta_l"]),
                n=int(row["n_switchers"]),
            )
        )
    for k in range(1, N_PLACEBOS + 1):
        row = detail.loc[-k]
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"Placebo_{k}",
                estimate=float(row["delta_l"]),
                n=int(row["n_switchers"]),
            )
        )

    rows.append(
        ParityRecord(
            module=MODULE,
            side="py",
            statistic="Av_tot_eff",
            estimate=float(fits["switchers"].estimate),
            n=int(len(df)),
        )
    )
    rows.append(
        ParityRecord(
            module=MODULE,
            side="py",
            statistic="Av_tot_eff_simple_weights",
            estimate=float(fits["simple"].estimate),
            n=int(len(df)),
        )
    )

    # --- switch-off design -------------------------------------------
    off = _switching_panel()
    dump_csv(off, f"{MODULE}_off")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        off_fit = sp.did_multiplegt_dyn(
            off,
            y="y",
            group="id",
            time="t",
            treatment="d",
            dynamic=OFF_EFFECTS - 1,
            placebo=OFF_PLACEBOS,
            cluster="id",
            n_boot=0,
        )
    off_es = off_fit.model_info["event_study"].set_index("relative_time")
    for k in range(1, OFF_EFFECTS + 1):
        row = off_es.loc[k - 1]
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"off_Effect_{k}",
                estimate=float(row["att"]),
                n=int(row["n_switchers"]),
            )
        )
    for k in range(1, OFF_PLACEBOS + 1):
        row = off_es.loc[-k]
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"off_Placebo_{k}",
                estimate=float(row["att"]),
                n=int(row["n_switchers"]),
            )
        )

    write_results(MODULE, "py", rows, extra={"estimator": "did_multiplegt_dyn"})


if __name__ == "__main__":
    main()
