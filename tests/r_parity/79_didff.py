"""StatsPAI functional-form test parity (Python side) -- Module 79.

Runs ``sp.functional_form_test`` on two panels; the companion R script runs
``didFF::didFF`` (Sant'Anna's own package, the reference implementation for
Roth & Sant'Anna 2023) on the same dumped CSVs.

Two designs are emitted on purpose:

* ``pt``  -- a log-scale DGP where the implied counterfactual density is
  non-negative everywhere and the test does not reject.
* ``rej`` -- a multiplicative DGP fed to a level-scale DiD, where the
  implied density goes negative on the bottom bin and the test rejects.

An accept-only fixture would leave the whole test-statistic path unexercised:
the p-value saturates at 1 whenever the max-t statistic is negative, so the
critical value could be arbitrarily wrong and the comparison would still pass.

Tolerance: rel < 1e-6 on every bin's implied density.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from _common import ParityRecord, dump_csv, write_results

MODULE = "79_didff"
N_BINS = 8


def _pt_panel() -> pd.DataFrame:
    """Additive DGP: parallel trends holds on the level scale it is run on."""
    rng = np.random.default_rng(31)
    rows = []
    for uid in range(1, 401):
        g = int(rng.choice([3, 0]))
        fe = rng.normal(6.0, 1.2)
        for t in range(1, 5):
            y = fe + 0.30 * t + (0.8 if (g > 0 and t >= g) else 0.0)
            y += rng.normal(0, 0.5)
            rows.append({"id": uid, "t": t, "g": g, "y": y})
    return pd.DataFrame(rows)


def _reject_panel() -> pd.DataFrame:
    """Multiplicative DGP read on the level scale.

    The treated group starts higher and everyone grows by the same *factor*,
    so parallel trends holds in logs and fails in levels. Run in levels, the
    design implies negative counterfactual mass at the bottom of the support.
    """
    rng = np.random.default_rng(5)
    rows = []
    for uid in range(1, 401):
        g = int(rng.choice([3, 0]))
        base = rng.lognormal(1.6 if g > 0 else 0.6, 0.35)
        for t in range(1, 5):
            y = base * (1.25 ** (t - 1)) * rng.lognormal(0.0, 0.10)
            rows.append({"id": uid, "t": t, "g": g, "y": y})
    return pd.DataFrame(rows)


def main() -> None:
    import statspai as sp

    rows: list[ParityRecord] = []
    for tag, frame in (("pt", _pt_panel()), ("rej", _reject_panel())):
        dump_csv(frame, f"{MODULE}_{tag}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.functional_form_test(
                frame,
                y="y",
                g="g",
                t="t",
                i="id",
                n_bins=N_BINS,
                n_sims=100_000,
                random_state=0,
            )
        for k, row in enumerate(res.table.itertuples(), start=1):
            rows.append(
                ParityRecord(
                    module=MODULE,
                    side="py",
                    statistic=f"{tag}_density_{k}",
                    estimate=float(row.implied_density),
                    n=int(res.n_units),
                )
            )
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"{tag}_pvalue",
                estimate=float(res.pvalue),
                n=int(res.n_units),
            )
        )

    write_results(MODULE, "py", rows, extra={"estimator": "functional_form_test"})


if __name__ == "__main__":
    main()
