"""StatsPAI two-way cluster-robust SE parity (Python side) -- Module 54.

Companion ``54_twoway_cluster.R`` runs ``lm`` + ``sandwich::vcovCL`` with a
two-way cluster formula on identical bytes.

Modules 14/15 pin the ONE-way cluster-robust SE. This module pins the
Cameron-Gelbach-Miller (2011) TWO-WAY cluster-robust SE -- ubiquitous in
empirical work (firm x time, state x year) and a known cross-implementation
pain point in its finite-sample correction.

``sp.twoway_cluster`` forms the inclusion-exclusion sandwich
``V = V1 + V2 - V12`` and applies the per-dimension Liang-Zeger correction
``G_i/(G_i-1) * (n-1)/(n-k)`` to each term. That convention matches
``sandwich::vcovCL(type="HC1", cadjust=TRUE)`` -- the package defaults and
the same reference module 14 uses for the one-way case -- to machine
precision (rel_se ~ 1e-9). ``fixest`` instead applies a single ``min(G)``
df factor and so differs at the ~1e-3 level; ``sandwich`` is the
like-for-like convention reference, so the headline is a strict machine-
precision pass rather than a documented convention gap.

DGP: a deterministic crossed two-way clustered DGP (seed=42), N=4000 with
40 x 50 crossed clusters, so both cluster dimensions carry enough levels
for a well-conditioned (positive-definite) two-way VCOV -- unlike a
low-cardinality dimension such as ``year`` in ``mpdta``, where the CGM
estimator degenerates. The committed ``data/54_twoway_cluster.csv`` is the
authoritative bytes the R side reads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "54_twoway_cluster"
FORMULA = "y ~ x"


def _make_dgp() -> pd.DataFrame:
    """Deterministic crossed two-way clustered DGP (NumPy PCG64, fixed seed)."""
    rng = np.random.default_rng(PARITY_SEED)
    n, g1_levels, g2_levels = 4000, 40, 50
    g1 = rng.integers(0, g1_levels, n)
    g2 = rng.integers(0, g2_levels, n)
    x = rng.normal(size=n) + 0.3 * g1 / g1_levels
    # Errors correlated within each cluster dimension -> genuine two-way
    # clustering that one-way SEs understate.
    u = rng.normal(size=n) + rng.normal(size=g1_levels)[g1] + rng.normal(size=g2_levels)[g2]
    y = 1.0 + 0.5 * x + u
    return pd.DataFrame({"y": y, "x": x, "g1": g1, "g2": g2})


def main() -> None:
    df = _make_dgp()
    dump_csv(df, MODULE)

    fit = sp.regress(FORMULA, data=df)
    tw = sp.twoway_cluster(fit, df, "g1", "g2")

    n = int(fit.data_info.get("n_obs", len(df)))
    rows: list[ParityRecord] = []
    for name in fit.params.index:
        beta = float(fit.params[name])
        canonical = "(Intercept)" if name == "Intercept" else name
        rows.append(
            ParityRecord(
                module=MODULE, side="py",
                statistic=f"beta_{canonical}",
                estimate=beta, se=float(tw.std_errors[name]), n=n,
            )
        )

    write_results(
        MODULE, "py", rows,
        extra={
            "formula": FORMULA,
            "vcov": "two-way cluster (Cameron-Gelbach-Miller 2011)",
            "cluster_vars": "g1 + g2",
            "r_reference": (
                "per-dimension Liang-Zeger G/(G-1)*(n-1)/(n-k) on "
                "V1+V2-V12; matches sandwich::vcovCL(type='HC1', "
                "cadjust=TRUE)"
            ),
        },
    )


if __name__ == "__main__":
    main()
