"""StatsPAI interflex parity (Python side) -- Module 87.

Runs the native ``sp.interflex`` port of Hainmueller, Mummolo and Xu's
interaction-effect estimators on a deterministic simulated sample with a
binary treatment, a moderator with a non-linear marginal-effect profile,
and one covariate, and emits:

- ``linear_me_<k>`` : marginal effect of D at evaluation point k with its
  HC1 delta-method SE (Y ~ X + D + D*X + Z);
- ``linear_ate``    : average treatment effect over treated observations
  and its delta-method SE;
- ``binning_x0_<j>``, ``binning_me_<j>`` : bin median and the bin-specific
  treatment effect (coefficient on D within bin j, HC1 SE) for three
  bins cut at the explicit cutoffs 0.3 and 1.7;
- ``lkurtosis``, ``p_wald``, ``p_lr`` : the moderator's L-kurtosis and the
  Wald / LR p-values of the linear-interaction restriction against the
  fully interacted binning model;
- ``kernel_me_<k>`` : the kernel (local linear, Gaussian, adaptive
  bandwidth ``bw = 1``) marginal effect at the same evaluation points.

The companion 87_interflex.R runs ``interflex::interflex`` with
``vartype = "delta"``, ``vcov.type = "robust"``, ``CI = TRUE`` and ``neval = 5``
(the same equally spaced grid); the Stata side runs the SSC ``interflex``
command on the same bytes.

Tolerance: rel_est 1e-6 and rel_se 1e-6 (closed-form OLS/WLS on both
sides; R's ``density()`` grid conventions are ported exactly so the
kernel rows are deterministic).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "87_interflex"
NEVAL = 5  # equally spaced from min(X) to max(X) -- the grid all three sides share
BW = 1.0
# Explicit bin cutoffs shared by all three sides: R's type-7 quantiles and
# Stata's percentile definition differ, so quantile-based bins would not be
# the same partition on the Stata side.
CUTOFFS = [0.3, 1.7]


def make_data(seed: int = PARITY_SEED, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(1.0, 1.2, n)
    z = rng.normal(0.0, 1.0, n)
    d = rng.binomial(1, 0.5, n).astype(float)
    # Mildly non-linear conditional marginal effect: 2 + 1.5 x - 0.25 x^2,
    # so the Wald / LR tests land in the informative range rather than at
    # a floating-point zero.
    me = 2.0 + 1.5 * x - 0.25 * x**2
    y = 1.0 + 0.8 * x + 0.5 * z + d * me + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"Y": y, "D": d, "X": x, "Z1": z})


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)
    n = int(len(df))
    rows: list[ParityRecord] = []

    lin = sp.interflex(df, y="Y", d="D", x="X", z=["Z1"], estimator="linear", neval=NEVAL)
    for k, (_, r) in enumerate(lin.detail.iterrows(), start=1):
        rows.append(ParityRecord(module=MODULE, side="py", statistic=f"linear_me_{k}",
                                 estimate=float(r["me"]), se=float(r["se"]), n=n))
    rows.append(ParityRecord(module=MODULE, side="py", statistic="linear_ate",
                             estimate=float(lin.estimate), se=float(lin.se), n=n))

    binned = sp.interflex(df, y="Y", d="D", x="X", z=["Z1"], estimator="binning", cutoffs=CUTOFFS)
    for j, (_, r) in enumerate(binned.detail.iterrows(), start=1):
        rows.append(ParityRecord(module=MODULE, side="py", statistic=f"binning_x0_{j}",
                                 estimate=float(r["x"]), n=int(r["n"])))
        rows.append(ParityRecord(module=MODULE, side="py", statistic=f"binning_me_{j}",
                                 estimate=float(r["me"]), se=float(r["se"]), n=int(r["n"])))
    tests = binned.model_info["tests"]
    rows.append(ParityRecord(module=MODULE, side="py", statistic="lkurtosis",
                             estimate=float(tests["x_lkurtosis"]), n=n))
    rows.append(ParityRecord(module=MODULE, side="py", statistic="p_wald",
                             estimate=float(tests["p_wald"]), n=n))
    rows.append(ParityRecord(module=MODULE, side="py", statistic="p_lr",
                             estimate=float(tests["p_lr"]), n=n))
    # Stata interflex's r(pwald): the fully interacted model leaves the
    # covariates uninteracted with the bins and the statistic is referred
    # to the F distribution (wald_full_moderate=False, wald_test="F").
    binned_stata = sp.interflex(df, y="Y", d="D", x="X", z=["Z1"], estimator="binning",
                                cutoffs=CUTOFFS, wald_full_moderate=False, wald_test="F")
    rows.append(ParityRecord(module=MODULE, side="py", statistic="p_wald_stata",
                             estimate=float(binned_stata.model_info["tests"]["p_wald"]), n=n))

    kern = sp.interflex(df, y="Y", d="D", x="X", z=["Z1"], estimator="kernel", bw=BW, neval=NEVAL)
    for k, (_, r) in enumerate(kern.detail.iterrows(), start=1):
        rows.append(ParityRecord(module=MODULE, side="py", statistic=f"kernel_me_{k}",
                                 estimate=float(r["me"]), n=n))

    # Stata interflex's kernel estimator uses the fixed Gaussian kernel
    # phi((X - x)/bw) with no density adaptation; the Stata side emits
    # these rows and R emits only the adaptive ones.
    kern_fixed = sp.interflex(df, y="Y", d="D", x="X", z=["Z1"], estimator="kernel", bw=BW, neval=NEVAL, adaptive=False)
    for k, (_, r) in enumerate(kern_fixed.detail.iterrows(), start=1):
        rows.append(ParityRecord(module=MODULE, side="py", statistic=f"kernel_fixed_me_{k}",
                                 estimate=float(r["me"]), n=n))

    write_results(
        MODULE, "py", rows,
        extra={
            "neval": NEVAL,
            "x_eval": "linspace(min(X), max(X), 5)",
            "cutoffs": CUTOFFS,
            "bw_kernel": BW,
            "vcov": "HC1 (interflex vcov.type='robust')",
            "density_port": (
                "kernel bandwidth adapts with R's stats::density (bw.nrd0, "
                "n = 512, cut = 3, linear binning + FFT, old.coords = FALSE), "
                "ported exactly"
            ),
            "wald_df": int(tests["df"]),
            "p_wald_rows": (
                "p_wald: R interflex (covariates interacted with the bins, "
                "chi-square); p_wald_stata: Stata interflex (covariates not "
                "interacted, F reference), reproduced by "
                "wald_full_moderate=False, wald_test='F'"
            ),
            "kernel_rows": (
                "kernel_me_<k>: adaptive bandwidth (R interflex); "
                "kernel_fixed_me_<k>: fixed Gaussian kernel (Stata interflex, adaptive=False)"
            ),
        },
    )


if __name__ == "__main__":
    main()
