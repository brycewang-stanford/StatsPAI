"""StatsPAI beta regression parity (Python side) -- Module 61.

Beta regression with a logit mean link and log-link precision, against
R betareg::betareg(..., link.phi='log') and Stata betareg (whose scale
equation also uses a log link), so all three sides estimate the same
(beta, log phi) parameterisation.
"""
from __future__ import annotations
import numpy as np, pandas as pd, statspai as sp
from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "61_betareg"


def make_data(n=900, seed=PARITY_SEED):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    mu = 1.0 / (1.0 + np.exp(-(0.5 + 0.6 * x1 - 0.8 * x2)))
    phi = 12.0
    y = rng.beta(mu * phi, (1.0 - mu) * phi)
    # Keep y strictly inside (0, 1) for all three references.
    y = np.clip(y, 1e-6, 1.0 - 1e-6)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def main():
    df = make_data()
    dump_csv(df, MODULE)
    res = sp.betareg(data=df, y="y", x=["x1", "x2"], link="logit", maxiter=2000, tol=1e-12)
    rows = []
    for nm, lab in [("_cons", "intercept"), ("x1", "x1"), ("x2", "x2")]:
        if nm in res.params.index:
            rows.append(ParityRecord(MODULE, "py", f"beta_{lab}",
                estimate=float(res.params[nm]),
                se=float(res.std_errors[nm]),
                n=int(len(df))))
    # Precision parameter on the common log scale.
    if "_cons_phi" in res.params.index:
        rows.append(ParityRecord(MODULE, "py", "ln_phi",
            estimate=float(res.params["_cons_phi"]),
            se=float(res.std_errors["_cons_phi"]),
            n=int(len(df))))
    write_results(MODULE, "py", rows,
                  extra={"link": "logit", "phi_link": "log"})


if __name__ == "__main__":
    main()
