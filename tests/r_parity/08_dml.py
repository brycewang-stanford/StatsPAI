"""StatsPAI DML PLR parity (Python side) -- Module 08.

Runs sp.dml(model='plr') with linear-regression nuisance learners
on the Card 1995 replica. Cross-fitting uses 5 folds with a fixed
seed. The companion 08_dml.R uses DoubleML::DoubleMLPLR with
mlr3::regr.lm and an identical fold split.

Tolerance: rel < 1e-3. With deterministic linear nuisance learners
the only Monte Carlo source is the fold split itself, so the gap
should be small.
"""
from __future__ import annotations

import numpy as np
import statspai as sp
from sklearn.linear_model import LinearRegression

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "08_dml"
COVARIATES = ["exper", "expersq", "black", "south", "smsa"]


def main() -> None:
    # simulated=True is load-bearing, not decoration. This module *writes*
    # data/08_dml.csv, and the committed fixture -- together with the R and
    # Stata goldens computed on it -- is the calibrated replica. The
    # loader's default later became the real extract, so a bare
    # card_1995() call silently regenerates the fixture from different
    # data and breaks the same-byte premise of the row. Original-data
    # Card parity is a separate ledger (tests/orig_parity/01_card_original).
    df = sp.datasets.card_1995(simulated=True)
    df["fold_id"] = np.arange(len(df)) % 5
    dump_csv(df, MODULE)

    # NOTE: pass numpy seed so sklearn's KFold split is deterministic;
    # sp.dml accepts learners through model_y / model_d kwargs.
    np.random.seed(PARITY_SEED)
    fit = sp.dml(
        data=df, y="lwage", d="educ", X=COVARIATES,
        model="plr",
        model_y=LinearRegression(),
        model_d=LinearRegression(),
        n_folds=5,
        fold_indices=df["fold_id"].to_numpy(),
    )

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="theta_DML_PLR",
            estimate=float(fit.estimate),
            se=float(fit.se),
            ci_lo=float(fit.ci[0]) if fit.ci is not None else None,
            ci_hi=float(fit.ci[1]) if fit.ci is not None else None,
            n=int(fit.n_obs),
        )
    ]

    write_results(
        MODULE, "py", rows,
        extra={
            "dml_model": "PLR",
            "n_folds": int(fit.model_info["n_folds"]),
            "ml_g": "LinearRegression",
            "ml_m": "LinearRegression",
            "seed": PARITY_SEED,
            "fold_source": str(fit.model_info["fold_source"]),
            "fold_column": "fold_id",
            "fold_parity_note": (
                "Python and R read the same fold_id column and use "
                "explicit sample-splitting APIs before fitting the "
                "linear-nuisance PLR score."
            ),
        },
    )


if __name__ == "__main__":
    main()
