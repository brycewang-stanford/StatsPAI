"""`sp.cardinality_match` solves its own program exactly.

Cardinality matching (Zubizarreta 2012) maximises the number of matched
units **subject to** a standardised-mean-difference tolerance on every
covariate.  The constraint is the whole point: the selected sample is only
useful because its balance is guaranteed by construction.

Until v1.22 StatsPAI relaxed the binary program to a continuous LP and
rounded the weights by a threshold.  Rounding does not preserve linear
constraints, so the returned sample routinely violated the very tolerance
the function advertises — and did so silently, with the balance table
reporting the breach as though it were fine.

There is no cross-package reference for this exact formulation.
``designmatch::cardmatch`` solves a *different* program (it selects both
arms; this one keeps every treated unit and chooses controls), so a
same-answer comparison would be meaningless.  The reference here is
therefore the **exact optimum of StatsPAI's own stated program**, solved
independently in this file, which is a stronger check than agreement with
another package solving something else: it pins both feasibility and
optimality.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import Bounds, LinearConstraint, milp

import statspai as sp

COVS = ["x1", "x2"]


def _dgp(seed: int, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-(0.8 * x1 - 0.4 * x2))))
    y = 2 * d + 0.7 * x1 + rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "d": d, "y": y})


def _fit(df, tol):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.cardinality_match(
            df, treatment="d", outcome="y", covariates=COVS, smd_tolerance=tol
        )


def _independent_optimum(df, tol):
    """Solve the documented program from scratch, in this file.

    maximise sum_j z_j
    s.t.  |mean(X_k | T=1) - sum_j z_j X_jk / sum_j z_j| <= tol * SD(X_k)
          sum_j z_j <= n_treated,   z_j in {0, 1}

    The denominator is ``X.std(axis=0)`` over the full sample — the same one
    the implementation and its balance table use, so the numbers below are
    directly comparable to `smd_tolerance`.
    """
    t = df["d"].to_numpy(dtype=int)
    X = df[COVS].to_numpy(dtype=float)
    ctrl = X[t == 0]
    mu_t = X[t == 1].mean(axis=0)
    sd_all = X.std(axis=0) + 1e-12
    tolv = tol * sd_all
    n_c, n_t = ctrl.shape[0], int((t == 1).sum())
    A = np.vstack(
        [
            (ctrl - (mu_t + tolv)).T,
            -(ctrl - (mu_t - tolv)).T,
            np.ones((1, n_c)),
        ]
    )
    b = np.concatenate([np.zeros(2 * len(COVS)), [float(n_t)]])
    res = milp(
        c=-np.ones(n_c),
        constraints=LinearConstraint(A, -np.inf, b),
        integrality=np.ones(n_c),
        bounds=Bounds(0, 1),
    )
    assert res.success, res.message
    z = np.round(res.x).astype(bool)
    mu_c = ctrl[z].mean(axis=0)
    return int(z.sum()), float(np.max(np.abs((mu_t - mu_c) / sd_all)))


GRID = [(seed, tol) for seed in range(6) for tol in (0.05, 0.1)]


@pytest.mark.parametrize("seed,tol", GRID)
class TestFeasibilityAndOptimality:
    def test_solution_satisfies_its_own_tolerance(self, seed, tol):
        """The guarantee the function sells. Was violated in 9 of these 12."""
        res = _fit(_dgp(seed), tol)
        assert float(res.balance["|SMD|"].max()) <= tol + 1e-9

    def test_matches_the_independent_optimum(self, seed, tol):
        """Exact, not just feasible: the same count an outside solve finds."""
        df = _dgp(seed)
        got = len(_fit(df, tol).control_matched)
        want, _ = _independent_optimum(df, tol)
        assert got == want

    def test_the_independent_optimum_is_itself_feasible(self, seed, tol):
        """Guards the oracle, so a bad formulation cannot excuse a bad fit."""
        _, smd = _independent_optimum(_dgp(seed), tol)
        assert smd <= tol + 1e-9


class TestMonotonicity:
    """Properties any correct solver must have; cheap regression tripwires."""

    def test_looser_tolerance_never_matches_fewer(self):
        df = _dgp(11)
        counts = [len(_fit(df, t).control_matched) for t in (0.02, 0.05, 0.1, 0.2)]
        assert counts == sorted(counts)

    def test_never_selects_more_controls_than_treated(self):
        df = _dgp(12)
        res = _fit(df, 0.5)
        assert len(res.control_matched) <= int((df["d"] == 1).sum())

    def test_pairs_are_one_to_one(self):
        res = _fit(_dgp(13), 0.1)
        assert len(res.treated_matched) == len(res.control_matched)
        assert len(set(res.treated_matched)) == len(res.treated_matched)
        assert len(set(res.control_matched)) == len(res.control_matched)

    def test_selected_units_are_controls(self):
        df = _dgp(14)
        res = _fit(df, 0.1)
        assert (df["d"].to_numpy()[res.control_matched] == 0).all()
        assert (df["d"].to_numpy()[res.treated_matched] == 1).all()


class TestInfeasibleRequest:
    def test_impossible_tolerance_fails_loudly(self):
        """An unsatisfiable program must raise, not return a breach."""
        df = _dgp(21)
        with pytest.raises(Exception) as exc:
            _fit(df, 0.0)
        assert "cardinality matching" in str(exc.value).lower() or isinstance(
            exc.value, (RuntimeError, ValueError)
        )
