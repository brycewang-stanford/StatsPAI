"""`SBWResult.solver_status` must report the solver, not the estimand.

The field was assigned the literal `"att"` / `"atc"` / `"ate"` — the
estimand — so a caller checking whether the optimiser converged got a
string that could never say. The real `scipy.optimize.minimize` outcome was
computed and discarded, including whether the loosened-`ftol` retry (which
exists precisely because SLSQP does fail here) had succeeded.

The estimand was never lost by fixing this: it is on `result.estimand` and
in `result.method`.

The returned weights are feasible either way — `_solve_sbw` verifies the
balance constraints and raises if they are violated, which is the check
`cardinality_match` was missing — so this is a diagnostics defect, not a
numerical one.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

COVS = ["x1", "x2"]


def _dgp(seed: int = 0, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-(0.7 * x1 - 0.7 * x2))))
    y = 2 * d + x1 + x2 + rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "d": d, "y": y})


def _fit(df, estimand="att", delta=0.02):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.sbw(
            df, treat="d", covariates=COVS, y="y", estimand=estimand, delta=delta
        )


@pytest.mark.parametrize("estimand", ["att", "atc", "ate"])
class TestSolverStatus:
    def test_reports_the_solver_not_the_estimand(self, estimand):
        res = _fit(_dgp(), estimand)
        assert res.solver_status != estimand
        assert res.solver_status == "optimal" or res.solver_status.startswith(
            "feasible-not-converged"
        )

    def test_estimand_is_still_recoverable(self, estimand):
        """The fix must not cost the information it displaced."""
        res = _fit(_dgp(), estimand)
        assert res.estimand.lower() == estimand
        assert estimand.upper() in res.method


class TestBalanceIsActuallyEnforced:
    """SBW's feasibility check is the pattern cardinality_match lacked."""

    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize("delta", [0.02, 0.005])
    def test_weights_satisfy_the_requested_tolerance(self, seed, delta):
        df = _dgp(seed)
        res = _fit(df, "att", delta)
        w = np.asarray(res.model_info["weights"], dtype=float)
        t = df["d"].to_numpy()
        X = df[COVS].to_numpy(dtype=float)
        sd = X.std(axis=0, ddof=1)
        mu_t = X[t == 1].mean(axis=0)
        wc = w[t == 0]
        mu_c = (X[t == 0] * wc[:, None]).sum(axis=0) / wc.sum()
        assert float(np.max(np.abs(mu_t - mu_c) / sd)) <= delta + 1e-6

    def test_exact_balance_is_reached_when_it_is_feasible(self):
        """delta=0 is not automatically infeasible.

        SBW at delta=0 asks for exact moment matching, which is attainable
        whenever the treated mean lies in the convex hull of the control
        covariates — the usual case. Asserting that delta=0 must raise would
        pin the wrong behaviour.
        """
        df = _dgp()
        res = _fit(df, "att", delta=0.0)
        w = np.asarray(res.model_info["weights"], dtype=float)
        t = df["d"].to_numpy()
        X = df[COVS].to_numpy(dtype=float)
        mu_t = X[t == 1].mean(axis=0)
        wc = w[t == 0]
        mu_c = (X[t == 0] * wc[:, None]).sum(axis=0) / wc.sum()
        assert float(np.max(np.abs(mu_t - mu_c) / X.std(axis=0, ddof=1))) < 1e-8

    def test_a_genuinely_infeasible_request_raises(self):
        """Target outside the control convex hull — no weighting reaches it."""
        df = _dgp()
        df.loc[df["d"] == 1, "x1"] = df.loc[df["d"] == 0, "x1"].max() + 5.0
        with pytest.raises(ValueError, match="infeasible"):
            _fit(df, "att", delta=0.0)
