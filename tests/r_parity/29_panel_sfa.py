"""StatsPAI panel SFA parity (Python side) -- Module 29.

Generates a deterministic time-invariant-inefficiency panel and
runs sp.xtfrontier(model='ti'). The companion 29_panel_sfa.R uses
sfaR::sfacross with id_var (Pitt-Lee 1981 time-invariant model).

All three sides fit the same half-normal Pitt-Lee (1981) model: R
frontier::sfa (truncNorm = FALSE) and Stata xtfrontier, ti with the
constraint [mu]_cons = 0 (its unconstrained ti model is the
truncated-normal Battese-Coelli 1988 model, a different likelihood).

Tolerance: rel_est 1e-3 (iterative). Standard errors: StatsPAI takes the
inverse of the central-difference observed-information Hessian; Stata's
xtfrontier uses an analytic Hessian (ml method d2) and agrees with it to
~3e-6 on every row, which pins the Python Hessian as exact. frontier's
mleCov (Coelli's FRONTIER 4.1 Fortran routine) differs by 0.1%-1.8%
across rows and is the loose side of the registered rel_se budget.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "29_panel_sfa"


def make_data(n_units: int = 50, T: int = 6, seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_units * T
    unit = np.repeat(np.arange(n_units), T)
    year = np.tile(np.arange(T), n_units)
    lnk = rng.normal(0, 1, n)
    lnl = rng.normal(0, 1, n)
    v = rng.normal(0, 0.3, n)
    # Time-invariant unit-level inefficiency
    unit_u = np.abs(rng.normal(0, 0.5, n_units))
    u = unit_u[unit]
    lny = 2.0 + 0.6 * lnk + 0.4 * lnl + v - u
    return pd.DataFrame({
        "lny": lny, "lnk": lnk, "lnl": lnl,
        "unit": unit, "year": year,
    })


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    fit = sp.xtfrontier(data=df, y="lny", x=["lnk", "lnl"],
                        id="unit", time="year",
                        model="ti", dist="half-normal")

    rows: list[ParityRecord] = [
        ParityRecord(MODULE, "py", "beta_intercept",
                     estimate=float(fit.params["_cons"]),
                     se=float(fit.std_errors["_cons"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "beta_lnk",
                     estimate=float(fit.params["lnk"]),
                     se=float(fit.std_errors["lnk"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "beta_lnl",
                     estimate=float(fit.params["lnl"]),
                     se=float(fit.std_errors["lnl"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "sigma_u",
                     estimate=float(fit.model_info["sigma_u"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "sigma_v",
                     estimate=float(fit.model_info["sigma_v"]),
                     n=int(len(df))),
    ]

    write_results(MODULE, "py", rows,
                  extra={"distribution": "half-normal",
                         "panel_model": "Pitt-Lee 1981 (ti)",
                         "reference_note": (
                             "Stata xtfrontier, ti is fit with the "
                             "constraint [mu]_cons = 0 so all three sides "
                             "share the half-normal likelihood; Stata's "
                             "analytic-Hessian SEs match StatsPAI to ~3e-6, "
                             "R frontier::sfa's mleCov differs by 0.1%-1.8%."
                         ),
                         "n_units": int(df["unit"].nunique()),
                         "T": int(df["year"].nunique())})


if __name__ == "__main__":
    main()
