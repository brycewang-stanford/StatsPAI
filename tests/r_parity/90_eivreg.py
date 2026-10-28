"""StatsPAI errors-in-variables regression parity (Python side) -- Module 90.

Pins ``sp.eivreg`` against Stata's ``eivreg`` (the canonical
implementation: the method has no R package by its authors) and against
R ``eivtools::eivreg`` 0.1-8, a port written to reproduce Stata's
estimator, as a bridge.

The estimator removes the assumed measurement-error variance from the
cross-product matrix, ``A = X'WX - diag((1 - r_j) sum_i w_i (x_ij -
xbar_j)^2)``, and solves ``A b = X'Wy``. Stata writes the correction as
``(N - 1) s_j^2`` and eivtools as ``N`` times the ML variance; both are
the centred sum of squares, so the point estimates are the same quantity.
The variance is the M-estimation sandwich with the variance-estimation
step stacked in (eivtools) or profiled out (Stata, StatsPAI); the
influence function of ``b`` is identical either way, so the standard
errors are the same quantity too, with no small-sample factor on any
side.

Two designs on one CSV: unweighted, and analytic weights ``[aw=w]``
(eivtools ``weights=``). Two error-prone regressors that are correlated
with each other, plus one regressor measured without error.

Tolerance: rel < 1e-6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "90_eivreg"

#: Reliability ratios var(true x) / var(observed x). Assumed, not
#: estimated: the regression treats them as known.
RELIABILITY = {"x1": 0.8, "x2": 0.7}

#: Coefficient order used for the row names (intercept last, as Stata).
TERMS = ("x1", "x2", "z", "Intercept")


def build_frame() -> pd.DataFrame:
    """Latent x1*, x2* (correlated), z exact; x observed with noise.

    The noise SDs (0.5, 0.6) put the true reliabilities near the assumed
    ones, so the corrected estimates are close to the truth (0.8, -0.5,
    0.3) and the correction is far from a no-op.
    """
    rng = np.random.default_rng(PARITY_SEED)
    n = 2000
    xs1 = rng.normal(size=n)
    xs2 = 0.5 * xs1 + rng.normal(size=n)
    z = rng.normal(size=n)
    x1 = xs1 + rng.normal(scale=0.5, size=n)
    x2 = xs2 + rng.normal(scale=0.6, size=n)
    y = 1.0 + 0.8 * xs1 - 0.5 * xs2 + 0.3 * z + rng.normal(size=n)
    w = rng.uniform(0.5, 2.0, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "z": z, "w": w})


def main() -> None:
    df = build_frame()
    dump_csv(df, MODULE)

    rows: list[ParityRecord] = []
    for design, weights in (("unw", None), ("aw", "w")):
        fit = sp.eivreg("y ~ x1 + x2 + z", df, reliability=RELIABILITY, weights=weights)
        for term in TERMS:
            name = "cons" if term == "Intercept" else term
            rows.append(
                ParityRecord(
                    module=MODULE,
                    side="py",
                    statistic=f"{design}_b_{name}",
                    estimate=float(fit.params[term]),
                    se=float(fit.std_errors[term]),
                    n=int(len(df)),
                )
            )

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "reliability": RELIABILITY,
            "designs": ["unweighted", "analytic weights [aw=w]"],
            "n_rows": len(rows),
            "reference": (
                "Stata 18 eivreg (canonical); R eivtools::eivreg 0.1-8 (bridge)"
            ),
        },
    )


if __name__ == "__main__":
    main()
