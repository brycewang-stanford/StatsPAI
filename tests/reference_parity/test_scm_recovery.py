"""Recovery-against-truth tests for the synthetic-control family.

These tests answer the reviewer-grade question raised against the
Basque-data parity rows (modules 07/18/19), whose cross-implementation
gaps are driven by SCM weight non-uniqueness (the V-optimisation is
non-convex) and, for the generalised/factor variants, by the fact that
a factor model is identified only up to rotation.  Parity against
another *implementation* is therefore the wrong yardstick for those
rows.  The right yardstick -- and the package's own strongest evidence
tier -- is recovery of a known estimand on a known data-generating
process:

  * ``test_classic_scm_unique_solution`` builds a DGP in which the
    treated unit is *exactly* a convex combination of the donors in the
    pre-period.  The convex weight problem then has a unique global
    minimiser, and ``sp.synth(method='classic')`` must recover the true
    weights and the true gap to numerical precision (the cross-language
    agreement with ``Synth::synth`` on the same bytes is checked in
    ``tests/r_parity/52_scm_unique``).

  * ``test_scm_family_recovers_known_att`` builds an interactive
    fixed-effects (factor) DGP with a known additive ATT and asserts
    that the classical, augmented, generalised (Xu 2017) and
    matrix-completion variants each recover that ATT within 3%.

Together these certify the estimators against truth, while the
Basque-data rows are retained in the cross-language harness purely as a
*documented* non-uniqueness disclosure, not as a strict parity claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# --------------------------------------------------------------------------- #
#  Unique-solution classical SCM
# --------------------------------------------------------------------------- #
def _unique_solution_dgp():
    """Treated unit is an exact convex combination of donors pre-period."""
    rng = np.random.default_rng(7)
    J, T0, T1 = 5, 20, 10
    T = T0 + T1
    t = np.arange(T)
    donors = np.zeros((J, T))
    for d in range(J):
        donors[d] = 5 + 0.3 * d + (1.0 + 0.2 * d) * np.sin(0.2 * t + d) + 0.05 * d * t
    w_true = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
    tau = 2.0
    treated = w_true @ donors
    treated[T0:] += tau  # known post-treatment effect
    rows = [(f"donor{d}", int(t[i]), donors[d, i]) for d in range(J) for i in range(T)]
    rows += [("treated", int(t[i]), treated[i]) for i in range(T)]
    df = pd.DataFrame(rows, columns=["region", "year", "y"])
    return df, w_true, tau, T0, J


def test_classic_scm_unique_solution():
    df, w_true, tau, T0, J = _unique_solution_dgp()
    fit = sp.synth(
        df,
        outcome="y",
        unit="region",
        time="year",
        treated_unit="treated",
        treatment_time=T0,
        method="classic",
    )

    # Exact weight recovery on the identified problem.
    wdf = fit.model_info["weights"]
    wmap = dict(zip(wdf["unit"], wdf["weight"]))
    w_hat = np.array([wmap.get(f"donor{d}", 0.0) for d in range(J)])
    assert np.allclose(
        w_hat, w_true, atol=1e-3
    ), f"weights {w_hat} differ from the unique solution {w_true}"

    # Perfect pre-fit and exact gap.
    assert fit.model_info["pre_treatment_rmse"] < 1e-6, (
        f"pre-treatment RMSE {fit.model_info['pre_treatment_rmse']:.2e} "
        f"should be ~0 on a convex-hull DGP"
    )
    assert (
        abs(float(fit.estimate) - tau) < 1e-3
    ), f"estimated gap {float(fit.estimate):.6f} != true tau {tau}"


# --------------------------------------------------------------------------- #
#  SCM family recovery on a factor DGP with a known ATT
# --------------------------------------------------------------------------- #
def _factor_dgp(tau: float = 4.0, noise: float = 0.05):
    rng = np.random.default_rng(20)
    N, T0, T1, r = 20, 20, 10, 2
    T = T0 + T1
    F = rng.normal(size=(T, r))
    L = rng.normal(size=(N, r))
    Y = L @ F.T + noise * rng.normal(size=(N, T))
    Y[0, T0:] += tau  # unit 0 treated, known ATT
    rows = [(f"u{i}", t, Y[i, t]) for i in range(N) for t in range(T)]
    df = pd.DataFrame(rows, columns=["unit", "time", "y"])
    return df, tau, T0


@pytest.mark.parametrize("method", ["classic", "augmented", "gsynth", "mc"])
def test_scm_family_recovers_known_att(method):
    df, tau, T0 = _factor_dgp()
    fit = sp.synth(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="u0",
        treatment_time=T0,
        method=method,
    )
    est = float(fit.estimate)
    rel = abs(est - tau) / abs(tau)
    assert rel < 0.03, (
        f"sp.synth(method='{method}') ATT={est:.4f} is {rel:.1%} from the "
        f"known factor-DGP ATT {tau} (>3%)"
    )
