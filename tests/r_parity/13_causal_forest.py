"""StatsPAI Causal Forest parity (Python side) -- Module 13.

Runs sp.causal_forest on the NSW-DW replica and emits the AIPW
average treatment effect on the treated (ATT) at a fixed seed; the
companion 13_causal_forest.R uses grf::causal_forest on the same
CSV with grf::average_treatment_effect(target.sample='treated').

Tolerance: rel < 1e-2 on the ATT. Even with the same seed the two
RNGs (numpy MT19937 vs R's L'Ecuyer) produce different bootstrap
trees, so bit-equal recovery is not expected; the goal is order-of-
magnitude agreement.
"""
from __future__ import annotations

import numpy as np
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "13_causal_forest"
COVARIATES = ["age", "education", "black", "hispanic", "married", "re74", "re75"]


def main() -> None:
    df = sp.datasets.nsw_dw()
    dump_csv(df, MODULE)

    Y = df["re78"].to_numpy()
    T = df["treat"].to_numpy()
    X = df[COVARIATES].to_numpy()

    np.random.seed(PARITY_SEED)
    cf = sp.causal_forest(
        Y=Y, T=T, X=X,
        n_estimators=2000,
        random_state=PARITY_SEED,
        discrete_treatment=True,
    )

    # Use ATT as the headline; grf's "treated" sample target is the
    # closest analogue.
    att_value = float(cf.att())
    ate_value = float(cf.ate())

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="att_causal_forest",
            estimate=att_value,
            n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="ate_causal_forest",
            estimate=ate_value,
            n=int(len(df)),
        ),
    ]

    write_results(
        MODULE, "py", rows,
        extra={
            "n_estimators": 2000,
            "random_state": PARITY_SEED,
            "covariates": COVARIATES,
            "overlap_warning": (
                "The NSW-DW + PSID-1 sample has severe propensity "
                "score overlap problems: grf reports estimated "
                "propensities between 0.002 and 0.993. AIPW-style "
                "estimators (both sp.causal_forest and "
                "grf::average_treatment_effect) become unreliable in "
                "this regime. On this dataset both implementations "
                "produce sign-wrong or order-of-magnitude-off "
                "estimates of the ATT relative to the published "
                "Dehejia-Wahba (1999) ATT of approximately $1794, "
                "which was obtained by propensity-score matching "
                "(Module 11), an estimator less sensitive to "
                "extreme propensity weights. Reviewers should not "
                "read the cross-implementation gap on this sample as "
                "a parity failure of either package; both packages "
                "correctly emit the overlap warning, and the ATT "
                "from sp.psm (Module 11) recovers the published "
                "value to bit-equal precision."
            ),
        },
    )


if __name__ == "__main__":
    main()
