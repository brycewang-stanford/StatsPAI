"""StatsPAI zero-inflated negative binomial parity (Python side) -- Module 64.

ZINB with a logit inflation equation, against R pscl::zeroinfl
(dist='negbin', link='logit') and Stata zinb ..., inflate(...).
Dispersion conventions differ across references (pscl reports theta,
Stata reports lnalpha with alpha = 1/theta), so the dispersion row is
exported on the common alpha scale as a point-estimate diagnostic.
"""
from __future__ import annotations
import numpy as np, pandas as pd, statspai as sp
from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "64_zinb"


def make_data(n=1500, seed=PARITY_SEED):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.binomial(1, 0.5, n).astype(float)
    z = rng.normal(0, 1, n)
    p_inflate = 1.0 / (1.0 + np.exp(-(-0.7 + 0.9 * z)))
    lam = np.exp(0.8 + 0.5 * x1 - 0.4 * x2)
    theta = 1.5
    # NB2 draw: gamma-mixed Poisson.
    g = rng.gamma(theta, lam / theta)
    counts = rng.poisson(g)
    infl = rng.binomial(1, p_inflate)
    y = np.where(infl == 1, 0, counts)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "z": z})


def main():
    df = make_data()
    dump_csv(df, MODULE)
    res = sp.zinb(formula="y ~ x1 + x2", data=df, inflate=["z"], maxiter=2000, tol=1e-12)
    name_map = [
        ("const", "count_intercept"),
        ("x1", "count_x1"),
        ("x2", "count_x2"),
        ("inflate_const", "inflate_intercept"),
        ("inflate_z", "inflate_z"),
    ]
    rows = []
    for nm, lab in name_map:
        if nm in res.params.index:
            rows.append(ParityRecord(MODULE, "py", f"beta_{lab}",
                estimate=float(res.params[nm]),
                se=float(res.std_errors[nm]),
                n=int(len(df))))
    # Dispersion on the common alpha = 1/theta scale (point estimate only).
    alpha = None
    for nm in ("alpha", "lnalpha", "ln_alpha", "theta"):
        if nm in res.params.index:
            val = float(res.params[nm])
            if nm == "alpha":
                alpha = val
            elif nm in ("lnalpha", "ln_alpha"):
                alpha = float(np.exp(val))
            else:
                alpha = 1.0 / val
            break
    if alpha is not None:
        rows.append(ParityRecord(MODULE, "py", "alpha", estimate=alpha,
                                 n=int(len(df))))
    write_results(MODULE, "py", rows,
                  extra={"inflate_link": "logit", "count_dist": "negbin",
                         "alpha_note": "alpha = 1/theta (Stata lnalpha scale)"})


if __name__ == "__main__":
    main()
