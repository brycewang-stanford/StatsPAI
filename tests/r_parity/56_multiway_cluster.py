"""StatsPAI three-way cluster-robust SE parity (Python side) -- Module 56.

Companion ``56_multiway_cluster.R`` runs ``lm`` + ``sandwich::vcovCL`` with a
three-way cluster formula on identical bytes.

Module 54 certifies the two-way wrapper ``sp.twoway_cluster``. This module
certifies the general n-way core ``sp.multiway_cluster_vcov`` at THREE-way,
which exercises the full Cameron-Gelbach-Miller inclusion-exclusion
(``V = ΣV_i - ΣV_ij + V_123``) including the triple-intersection term.

``sp.multiway_cluster_vcov`` applies the per-dimension Liang-Zeger CR1
correction to each inclusion-exclusion component, matching
``sandwich::vcovCL(cluster = ~ g1 + g2 + g3, type="HC1", cadjust=TRUE)`` to
machine precision (rel_se ~ 1e-7). This is the regression guard for the
v1.16.1 correctness fix: the intersection cluster key previously collided
under a ``"\\0"`` string join (NumPy strips the NUL), undercounting the
intersection cells and biasing the multiway VCOV (see MIGRATION.md).

DGP: a deterministic crossed three-way clustered DGP (seed=42), N=4000 with
40 x 50 x 30 crossed clusters, so all three dimensions and their pairwise /
triple intersections carry enough levels for a well-conditioned VCOV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "56_multiway_cluster"
FORMULA = "y ~ x"


def _make_dgp() -> pd.DataFrame:
    """Deterministic crossed three-way clustered DGP (NumPy PCG64, fixed seed)."""
    rng = np.random.default_rng(PARITY_SEED)
    n, g1_levels, g2_levels, g3_levels = 4000, 40, 50, 30
    g1 = rng.integers(0, g1_levels, n)
    g2 = rng.integers(0, g2_levels, n)
    g3 = rng.integers(0, g3_levels, n)
    x = rng.normal(size=n) + 0.3 * g1 / g1_levels
    u = (
        rng.normal(size=n)
        + rng.normal(size=g1_levels)[g1]
        + rng.normal(size=g2_levels)[g2]
        + rng.normal(size=g3_levels)[g3]
    )
    y = 1.0 + 0.5 * x + u
    return pd.DataFrame({"y": y, "x": x, "g1": g1, "g2": g2, "g3": g3})


def main() -> None:
    df = _make_dgp()
    dump_csv(df, MODULE)

    fit = sp.regress(FORMULA, data=df)
    beta = fit.params.values
    X = np.column_stack([np.ones(len(df)), df["x"].to_numpy(float)])
    resid = df["y"].to_numpy(float) - X @ beta
    clusters = [df["g1"].to_numpy(), df["g2"].to_numpy(), df["g3"].to_numpy()]
    se = np.sqrt(np.diag(sp.multiway_cluster_vcov(X, resid, clusters)))

    n = int(fit.data_info.get("n_obs", len(df)))
    rows: list[ParityRecord] = []
    for i, name in enumerate(fit.params.index):
        b = float(fit.params[name])
        canonical = "(Intercept)" if name == "Intercept" else name
        rows.append(
            ParityRecord(
                module=MODULE, side="py",
                statistic=f"beta_{canonical}",
                estimate=b, se=float(se[i]), n=n,
            )
        )

    write_results(
        MODULE, "py", rows,
        extra={
            "formula": FORMULA,
            "vcov": "three-way cluster (Cameron-Gelbach-Miller 2011)",
            "cluster_vars": "g1 + g2 + g3",
            "matches": "sandwich::vcovCL(cluster=~g1+g2+g3, type='HC1', cadjust=TRUE)",
        },
    )


if __name__ == "__main__":
    main()
