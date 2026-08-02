"""StatsPAI dCDH 2020 DID_M parity (Python side) -- Module 81.

Runs ``sp.did_multiplegt`` on a non-absorbing panel where treatment switches
both on and off; the companion R script runs ``DIDmultiplegt::did_multiplegt``
on the same dumped CSV.

Version note that makes this module possible at all: the CRAN package's 2.x
rewrite routes the classic estimator through ``mode="old"``, and that path
returns ``NaN`` even on the package's own bundled example. The archived
**0.1.4** is the last release where it works, and it is what this pins
against. See the R script for the install line.

Emitted: the static DID_M effect, the dynamic effect at horizon 1, and the
placebo at lag 1 -- the three the reference reports for this call. The
placebo is requested with ``placebo_sign="r"``: dCDH's own Stata and R
implementations disagree on its sign, and this module compares against R. Standard
errors are not compared: the R side runs ``brep=0`` and StatsPAI's come from
its own bootstrap.

Tolerance: rel < 1e-6.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from _common import ParityRecord, dump_csv, write_results

MODULE = "81_didm"


def _panel() -> pd.DataFrame:
    """Treatment switches on AND off -- the design DID_M exists for.

    An absorbing panel would leave the switch-off branch and its sign
    convention untested, which is where this estimator differs from every
    cohort-based one.
    """
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.did_multiplegt(
            df,
            y="y",
            group="id",
            time="t",
            treatment="d",
            placebo=1,
            dynamic=1,
            n_boot=0,
            # dCDH's Stata and R implementations disagree on the placebo's
            # SIGN -- on did::mpdta both give |0.024269| and the three
            # effects agree to six decimals, but Stata reports it positive
            # and DIDmultiplegt negative. This module compares against R,
            # so it asks for R's convention; the default stays on Stata's.
            placebo_sign="r",
        )
    es = res.model_info["event_study"].set_index("relative_time")["att"]

    rows = [
        ParityRecord(
            module=MODULE,
            side="py",
            statistic="effect",
            estimate=float(es.loc[0]),
            n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE,
            side="py",
            statistic="dynamic_1",
            estimate=float(es.loc[1]),
            n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE,
            side="py",
            statistic="placebo_1",
            estimate=float(es.loc[-1]),
            n=int(len(df)),
        ),
    ]
    write_results(MODULE, "py", rows, extra={"estimator": "did_multiplegt"})


if __name__ == "__main__":
    main()
