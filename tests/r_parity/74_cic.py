"""StatsPAI Changes-in-Changes parity (Python side) -- Module 74.

Runs ``sp.cic`` on a two-period panel; the companion R script runs
``qte::CiC`` on the same dumped CSV.

``sp.cic`` splits the four (group x time) cells on ``t == 0`` / ``t == 1``,
while ``qte::CiC`` takes explicit ``t`` / ``tmin1`` values, so the CSV keeps
the raw 1/2 coding and this side derives ``post``.

Tolerance: rel < 1e-6 on the ATT and on each quantile treatment effect.
Standard errors are not compared -- the R call runs with ``se = FALSE`` and
StatsPAI's come from its own bootstrap.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from _common import ParityRecord, dump_csv, write_results

MODULE = "74_cic"
PROBS = [round(0.1 * k, 1) for k in range(1, 10)]


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    n = 600
    treat = rng.integers(0, 2, n)
    ui = rng.normal(0, 1, n)
    rows = []
    for i in range(n):
        for t in (1, 2):
            y = ui[i] + 0.5 * t + 2.0 * (treat[i] * (t == 2)) + rng.normal(0, 1)
            rows.append({"id": i + 1, "t": t, "treat": int(treat[i]), "y": y})
    return pd.DataFrame(rows)


def main() -> None:
    import statspai as sp

    df = _panel()
    dump_csv(df, MODULE)

    work = df.copy()
    work["post"] = (work["t"] == 2).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.cic(
            work,
            y="y",
            group="treat",
            time="post",
            quantiles=PROBS,
            n_boot=1,
            seed=0,
        )

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE,
            side="py",
            statistic="cic_ATT",
            estimate=float(fit.estimate),
            se=None,
            n=int(len(df)),
        )
    ]
    detail = fit.detail.set_index("quantile")
    for p in PROBS:
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"qte_{round(p * 100):02d}",
                estimate=float(detail.loc[p, "qte"]),
                se=None,
                n=int(len(df)),
            )
        )

    write_results(MODULE, "py", rows, extra={"estimator": "cic"})


if __name__ == "__main__":
    main()
