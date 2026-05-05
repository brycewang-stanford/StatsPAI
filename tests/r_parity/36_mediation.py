"""StatsPAI causal mediation parity (Python side) -- Module 36.

Generates a deterministic mediator-mediated DGP and runs sp.mediation.
The companion 36_mediation.R uses mediation::mediate.

Tolerance: rel < 5e-2 on ACME/ADE/total_effect (mediation analysis
involves OLS-based decomposition + bootstrap; both implementations
share the Imai-Keele-Tingley framework).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "36_mediation"


def make_data(n: int = 800, seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    treat = rng.binomial(1, 0.5, n)
    m = 0.3 * treat + rng.normal(0, 0.5, n)
    y = 0.5 * treat + 0.4 * m + rng.normal(0, 0.4, n)
    return pd.DataFrame({"treat": treat, "m": m, "y": y})


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    res = sp.mediation(df, y="y", d="treat", m="m")
    mi = res.model_info

    rows: list[ParityRecord] = [
        ParityRecord(MODULE, "py", "acme",
                     estimate=float(mi["acme"]),
                     se=float(mi["se_acme"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "ade",
                     estimate=float(mi["ade"]),
                     se=float(mi["se_ade"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "total_effect",
                     estimate=float(mi["total_effect"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "prop_mediated",
                     estimate=float(mi["prop_mediated"]),
                     n=int(len(df))),
    ]

    write_results(MODULE, "py", rows,
                  extra={"n_boot": int(mi["n_boot"])})


if __name__ == "__main__":
    main()
