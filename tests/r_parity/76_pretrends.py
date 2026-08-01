"""StatsPAI pre-trends power parity (Python side) -- Module 76.

Compares ``sp.pretrends_power`` and ``sp.pretrends_slope_for_power`` against
Roth's ``pretrends`` R package on a hand-crafted event study, so the R side
can mirror the inputs without an intermediate CSV.

The estimator is a function of ``(betahat, sigma)`` alone -- no panel is
involved -- so both sides build the same covariance from the same literal
standard errors and an AR(1)-style correlation, which keeps the two inputs
bit-identical.

Tolerance: rel < 1e-3. This module sits in the iterative tier on purpose.
``pretrends`` gets its rejection probability from ``mvtnorm::pmvnorm``,
whose Genz-Bretz integrator is randomised: twenty repeated calls on this
fixture spread over ~5e-4 (sd 1.3e-4). A tighter bound would be pinning R's
Monte-Carlo noise, not StatsPAI's answer. The likelihood ratio is the
exception -- it is a closed-form multivariate normal density ratio and
agrees to ~1e-11.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from _common import ParityRecord, write_results

MODULE = "76_pretrends"

# Event time; -1 is the omitted reference period.
T_VEC = [-4.0, -3.0, -2.0, 0.0, 1.0, 2.0]
SE = [0.050, 0.045, 0.040, 0.100, 0.110, 0.120]
BETA = [0.012, -0.008, 0.021, 0.180, 0.240, 0.310]
RHO = 0.5  # corr(i, j) = RHO ** |i - j|
SLOPES = [0.02, 0.05]
TARGET_POWERS = [0.5, 0.8]


def _sigma() -> np.ndarray:
    k = len(SE)
    idx = np.arange(k)
    corr = RHO ** np.abs(idx[:, None] - idx[None, :])
    d = np.asarray(SE, dtype=float)
    return corr * np.outer(d, d)


def main() -> None:
    import statspai as sp
    from statspai.core.results import CausalResult

    sigma = _sigma()
    t = np.asarray(T_VEC, dtype=float)
    pre = t < -1

    es = pd.DataFrame({"relative_time": t, "att": BETA, "se": SE})
    res = CausalResult(
        method="ParityPretrendsInput",
        estimand="ATT(0)",
        estimate=BETA[3],
        se=SE[3],
        pvalue=0.0,
        ci=(BETA[3] - 1.96 * SE[3], BETA[3] + 1.96 * SE[3]),
        alpha=0.05,
        n_obs=1000,
        model_info={"event_study": es, "vcv_pre": sigma[np.ix_(pre, pre)]},
    )

    rows: list[ParityRecord] = []
    for slope in SLOPES:
        tag = f"{slope:g}".replace(".", "p")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = sp.pretrends_power(res, delta=slope * (t[pre] + 1.0))
        for key in ("power", "bayes_factor", "likelihood_ratio"):
            rows.append(
                ParityRecord(
                    module=MODULE,
                    side="py",
                    statistic=f"{key}_slope_{tag}",
                    estimate=float(out[key]),
                    n=1000,
                )
            )

    for target in TARGET_POWERS:
        tag = f"{target:g}".replace(".", "p")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = sp.pretrends_slope_for_power(res, target_power=target)
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"slope_for_power_{tag}",
                estimate=float(out["slope"]),
                n=1000,
            )
        )

    write_results(MODULE, "py", rows, extra={"estimator": "pretrends_power"})


if __name__ == "__main__":
    main()
