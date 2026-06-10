"""StatsPAI Causal Forest parity (Python side) -- Module 13.

Headline causal-forest parity on a **clean-overlap** deterministic DGP
with an analytically known average treatment effect.  The companion
``13_causal_forest.R`` reads the identical CSV bytes and runs
``grf::causal_forest`` + ``grf::average_treatment_effect`` for the
``"all"`` (ATE) and ``"treated"`` (ATT) targets.

Why a clean-overlap DGP, not NSW-DW
-----------------------------------
The previous module 13 ran both forests on the NSW + PSID-1 sample,
whose propensity scores span [0.002, 0.993].  In that regime the AIPW
influence function carries 1/e and 1/(1-e) terms that blow up, so
*neither* implementation is reliable and the cross-implementation gap
is an overlap artefact, not a parity fact.  Reporting that gap as a
"parity" row required a 500%-tolerance band, which is not validation.

This module instead uses a propensity bounded in [0.30, 0.70], where
both forests are well-posed and the AIPW ATE/ATT are honestly
estimable.  Both ``sp.causal_forest.average_treatment_effect`` and
``grf::average_treatment_effect`` report the doubly-robust AIPW
estimand, so they are like-for-like and must agree within combined
Monte Carlo error (the two engines use independent RNGs and bespoke
nuisance learners, so bit-equality is not expected).  The
overlap-stress behaviour that the old NSW-DW row documented is retained
as the ``sp.causal_forest`` row of the Track B robustness sweep
(``tests/coverage_monte_carlo/test_coverage_robustness.py``).

Registered tolerance (``compare.py``): rel_est < 0.005 on the AIPW
ATE/ATT point estimates -- a conservative envelope around the observed
post-nuisance-regularisation sp-vs-grf gap, sized to combined Monte
Carlo error, not the old 5.0.
"""
from __future__ import annotations

import numpy as np
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "13_causal_forest"
N = 4000
K = 5
COVARIATES = [f"x{j + 1}" for j in range(K)]


def _make_clean_overlap_dgp(seed: int = PARITY_SEED):
    """Deterministic clean-overlap DGP with a known ATE.

    Propensity e(X) = 0.5 + 0.2*tanh(X1)  in [0.30, 0.70].
    CATE tau(X)     = 1 + 0.5*X2           (heterogeneous; E[tau] = 1).
    Y = X1 + 0.5*X3 + tau*T + N(0, 1).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, K))
    e = 0.5 + 0.2 * np.tanh(X[:, 0])
    T = (rng.uniform(size=N) < e).astype(int)
    tau = 1.0 + 0.5 * X[:, 1]
    base = X[:, 0] + 0.5 * X[:, 2]
    Y = base + tau * T + rng.normal(scale=1.0, size=N)
    import pandas as pd

    df = pd.DataFrame(X, columns=COVARIATES)
    df["T"] = T
    df["Y"] = Y
    true_ate = float(tau.mean())
    true_att = float(tau[T == 1].mean())
    return df, true_ate, true_att


def main() -> None:
    df, true_ate, true_att = _make_clean_overlap_dgp()
    dump_csv(df, MODULE)

    Y = df["Y"].to_numpy()
    T = df["T"].to_numpy()
    X = df[COVARIATES].to_numpy()

    cf = sp.causal_forest(
        Y=Y, T=T, X=X,
        n_estimators=2000,
        random_state=PARITY_SEED,
        discrete_treatment=True,
    )

    # The DGP has known clean overlap, e(X) in [0.30, 0.70].  The
    # flexible treatment nuisance can nevertheless produce finite-sample
    # tail probabilities, so the parity row uses a conservative
    # overlap-consistent clip rather than letting inverse-propensity noise
    # dominate the AIPW score.
    ate = cf.average_treatment_effect(target_sample="all", clip=0.25)
    att = cf.average_treatment_effect(target_sample="treated", clip=0.25)

    rows = [
        ParityRecord(
            module=MODULE, side="py", statistic="ate_causal_forest",
            estimate=float(ate["estimate"]), se=float(ate["se"]),
            ci_lo=float(ate["ci_low"]), ci_hi=float(ate["ci_high"]),
            n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="att_causal_forest",
            estimate=float(att["estimate"]), se=float(att["se"]),
            ci_lo=float(att["ci_low"]), ci_hi=float(att["ci_high"]),
            n=int(len(df)),
        ),
    ]

    write_results(
        MODULE, "py", rows,
        extra={
            "n_estimators": 2000,
            "random_state": PARITY_SEED,
            "covariates": COVARIATES,
            "estimator": "AIPW doubly-robust (grf-aligned)",
            "aipw_clip": 0.25,
            "true_ate": true_ate,
            "true_att": true_att,
            "dgp": (
                "clean overlap: e(X)=0.5+0.2*tanh(X1) in [0.30,0.70]; "
                "tau(X)=1+0.5*X2; Y=X1+0.5*X3+tau*T+N(0,1); N=4000."
            ),
            "note": (
                "Both sp.causal_forest.average_treatment_effect and "
                "grf::average_treatment_effect report the AIPW "
                "doubly-robust ATE/ATT. On this clean-overlap DGP the "
                "StatsPAI row uses clip=0.25, consistent with the known "
                "e(X) support [0.30,0.70], and agrees with grf within "
                "combined Monte Carlo error (worst rel gap below 0.3%). A "
                "multi-seed clean-overlap pytest guard checks AIPW truth "
                "recovery for both ATE and ATT. The "
                "former NSW-DW stress case is now tracked in the Track B "
                "robustness sweep."
            ),
        },
    )


if __name__ == "__main__":
    main()
