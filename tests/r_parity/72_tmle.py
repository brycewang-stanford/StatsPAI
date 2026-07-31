"""StatsPAI TMLE parity -- Module 72 (Python side).

Pins ``sp.tmle`` against ``tmle::tmle`` (Gruber & van der Laan), the
reference implementation of targeted maximum likelihood estimation.

Why this module exists
----------------------
``sp.tmle`` was already graded ``bit-exact``, but against a *frozen
base-R fixture*: a hand-rolled ``stats::glm`` TMLE written for the
occasion. That certifies the arithmetic against a reference StatsPAI
itself specified, which is a weaker claim than agreeing with the
package the method's authors maintain. This module replaces the anchor
with the real one. The frozen-glm fixture is kept as a second,
independent line of evidence (``test_tmle_parity.py``).

Shared-nuisance design
----------------------
TMLE has two stages: an initial fit of ``Q(a, W) = E[Y | A=a, W]`` and
``g(W) = P(A=1 | W)``, then a targeting step that fluctuates ``Q`` along
the least-favourable submodel. Only the second stage is the estimator;
the first is whatever learner the user supplies. Both ``tmle::tmle`` and
(since v1.21) ``sp.tmle`` accept ``Q`` and ``g1W`` directly, so this
module computes them once on the Python side -- unpenalised logistic
MLEs -- ships them through the CSV, and lets each engine run only its
targeting step on identical inputs.

The outcome is **binary** on purpose. For continuous outcomes both
implementations rescale ``Y`` to [0, 1] for the logistic fluctuation and
map back afterwards; a binary outcome removes that transformation from
the comparison so the row grades the targeting step alone.

Fluctuation convention -- the reason ``fluctuation='per_arm'`` exists
--------------------------------------------------------------------
The two packages target along **different submodels**:

* StatsPAI's default (``fluctuation='single'``) uses one clever
  covariate ``H(A,W) = A/g - (1-A)/(1-g)`` and a scalar ``epsilon``
  (van der Laan & Rubin 2006).
* ``tmle::tmle`` uses **two** clever covariates, ``A/g`` and
  ``-(1-A)/(1-g)``, fitted jointly -- its ``$epsilon`` is a 2-vector.

Both are valid TMLEs solving the efficient-influence-function equation
and are asymptotically equivalent, but they differ at finite ``n``: on
this fixture the two psi values differ in the third decimal. That is a
convention gap, not a defect, so StatsPAI keeps ``'single'`` as its
documented default and this module pins the ``'per_arm'`` mode, which
reproduces ``tmle::tmle`` to ~1e-11. The ``'single'`` value is recorded
in the JSON ``extra`` block so the size of the gap stays visible rather
than being quietly dropped.

Propensity truncation is disabled on both sides (``gbound = 1e-8``) --
the DGP keeps ``g`` inside (0.25, 0.75), so no unit is near a bound and
the parameter would only introduce a spurious convention difference.

Registered tolerance (``compare.py``): rel_est < 1e-6 (machine tier).
The residual is Newton-iteration tolerance on the fluctuation
parameters, observed at ~2e-12.

References
----------
[@vanderlaan2006targeted], [@gruber2012tmle]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp
from sklearn.linear_model import LogisticRegression

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "72_tmle"
N = 1200
K = 3
COVARIATES = [f"w{j + 1}" for j in range(K)]
GBOUND = 1e-8


def make_data(seed: int = PARITY_SEED) -> pd.DataFrame:
    """Binary-outcome DGP with propensities bounded well inside (0, 1)."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(N, K))
    # g(W) = 0.5 + 0.25*tanh(W1) lies in (0.25, 0.75): truncation inert.
    g = 0.5 + 0.25 * np.tanh(W[:, 0])
    A = (rng.uniform(size=N) < g).astype(int)
    p_y = 1.0 / (
        1.0 + np.exp(-(-0.3 + 0.8 * A + 0.6 * W[:, 0] - 0.4 * W[:, 1]))
    )
    Y = (rng.uniform(size=N) < p_y).astype(int)

    df = pd.DataFrame(W, columns=COVARIATES)
    df["A"] = A
    df["Y"] = Y

    # Initial fits, computed once and shared with R through the CSV.
    lr_q = LogisticRegression(penalty=None, max_iter=5000, tol=1e-10)
    lr_q.fit(np.column_stack([A, W]), Y)
    df["Q0"] = lr_q.predict_proba(np.column_stack([np.zeros(N), W]))[:, 1]
    df["Q1"] = lr_q.predict_proba(np.column_stack([np.ones(N), W]))[:, 1]

    lr_g = LogisticRegression(penalty=None, max_iter=5000, tol=1e-10)
    lr_g.fit(W, A)
    df["g1W"] = lr_g.predict_proba(W)[:, 1]
    return df


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    Q = df[["Q0", "Q1"]].to_numpy()
    g1W = df["g1W"].to_numpy()
    common = dict(
        data=df,
        y="Y",
        treat="A",
        covariates=COVARIATES,
        Q=Q,
        g1W=g1W,
        propensity_bounds=(GBOUND, 1.0 - GBOUND),
    )

    per_arm = sp.tmle(fluctuation="per_arm", **common)
    single = sp.tmle(fluctuation="single", **common)

    info = per_arm.model_info
    assert info["nuisance_source"] == {"Q": "supplied", "g1W": "supplied"}, info

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="psi_tmle_ate",
            estimate=float(per_arm.estimate), se=float(per_arm.se),
            ci_lo=float(per_arm.ci[0]) if per_arm.ci is not None else None,
            ci_hi=float(per_arm.ci[1]) if per_arm.ci is not None else None,
            n=N,
        )
    ]

    write_results(
        MODULE, "py", rows,
        extra={
            "seed": PARITY_SEED,
            "covariates": COVARIATES,
            "outcome_type": "binary",
            "fluctuation": "per_arm",
            "gbound": GBOUND,
            "epsilon_vec": info["epsilon_vec"],
            "epsilon_basis_note": (
                "The two engines report epsilon in different bases, so the "
                "vectors are not elementwise comparable and are deliberately "
                "not emitted as parity rows. StatsPAI fits on the columns "
                "[A/g, -(1-A)/(1-g)]; tmle::tmle fits on "
                "[(1-A)/(1-g), A/g]. The two spans -- and therefore the "
                "fluctuated Q* and psi -- are identical, which is what the "
                "psi_tmle_ate row checks; the coefficient vectors differ "
                "only by that reordering and sign flip."
            ),
            "nuisance": (
                "Q(0,W) / Q(1,W) / g1W are unpenalised logistic MLEs "
                "computed once on the Python side and shipped to R in the "
                "CSV, so only the targeting step differs between engines."
            ),
            "single_fluctuation_psi": float(single.estimate),
            "single_fluctuation_se": float(single.se),
            "convention_note": (
                "StatsPAI's documented default fluctuation='single' uses "
                "one clever covariate H = A/g - (1-A)/(1-g) and a scalar "
                "epsilon (van der Laan & Rubin 2006); tmle::tmle fluctuates "
                "along two per-arm covariates and reports a 2-vector "
                "epsilon. Both solve the EIF equation and are "
                "asymptotically equivalent, but differ at finite n: on this "
                "fixture psi is "
                f"{float(single.estimate):.9f} under 'single' versus "
                f"{float(per_arm.estimate):.9f} under 'per_arm' "
                "(rel gap ~1.3e-3). The parity row pins 'per_arm', the "
                "convention tmle::tmle uses; the 'single' value is recorded "
                "here so the size of the gap stays visible."
            ),
        },
    )


if __name__ == "__main__":
    main()
