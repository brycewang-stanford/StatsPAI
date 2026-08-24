"""Stata ``did_imputation, project()`` parity.

``project(varlist)`` regresses the imputed per-observation treatment
effects on covariates and reports the constant and slopes — the
continuous counterpart of ``hetby``, which splits into cells. StatsPAI
had neither the option nor any way to reach the underlying effects.

Golden numbers: Stata 18 MP, ``did_imputation`` (2023-11-22) on ``mpdta``,
``did_imputation lemp countyreal year Ei, project(lpop)``. Generating
do-file ``tests/stata_parity/78_bjs_project.do``.

Tolerances, and why they differ between coefficients and SEs
------------------------------------------------------------
**Coefficients: 1e-6.** Observed worst case 4.2e-7, from the sparse
``lsqr`` Y(0) fit versus reghdfe's direct absorb.

**SEs: 5% relative.** StatsPAI's analytic influence-function SE for the
imputation estimator is *already known* to disagree with Stata — it
under-counts the variance contributed by estimating the unit/time fixed
effects, is anti-conservative at roughly 0.87 coverage, and the estimator
emits a ``UserWarning`` pointing at ``vce='bootstrap'``. The projection
inherits exactly that influence function, so it inherits exactly that
gap (2.8% on the constant, 1.8% on the slope here).

That inheritance is not assumed — ``test_projection_reduces_to_the_att``
pins it: projecting on the constant alone reproduces the estimator's own
ATT and its analytic SE to floating-point noise. So a *change* in the
SE gap means the shared influence function moved, which is the thing
worth catching; the absolute offset is pre-existing and tracked
separately.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

import statspai as sp

_MPDTA = (
    pathlib.Path(__file__).resolve().parents[1]
    / "orig_parity"
    / "data"
    / "02_mpdta_original.csv"
)

KEYS = dict(y="lemp", group="countyreal", time="year", first_treat="first_treat")

COEF_ATOL = 1e-6
SE_RTOL = 0.05

# did_imputation lemp countyreal year Ei, project(lpop)
STATA_PROJECT_LPOP = {
    "_cons": (-0.113562670701214, 0.039880486206007),
    "lpop": (0.018577544710297, 0.009359473656994),
}


@pytest.fixture
def mpdta() -> pd.DataFrame:
    return pd.read_csv(_MPDTA)


class TestProjectParity:
    def test_project_coefficients_match_stata(self, mpdta):
        res = sp.did_imputation(mpdta, **KEYS, project=["lpop"])
        got = {r.term: r.coef for r in res.diagnostics["project"].itertuples()}
        assert set(got) == set(STATA_PROJECT_LPOP)
        for term, (coef, _) in STATA_PROJECT_LPOP.items():
            assert got[term] == pytest.approx(
                coef, abs=COEF_ATOL
            ), f"{term}: StatsPAI {got[term]:.12f} vs Stata {coef:.12f}"

    def test_project_ses_match_stata_within_the_known_band(self, mpdta):
        res = sp.did_imputation(mpdta, **KEYS, project=["lpop"])
        got = {r.term: r.se for r in res.diagnostics["project"].itertuples()}
        for term, (_, se) in STATA_PROJECT_LPOP.items():
            assert got[term] == pytest.approx(
                se, rel=SE_RTOL
            ), f"{term}: SE {got[term]:.12f} vs Stata {se:.12f}"

    def test_projection_reduces_to_the_att(self, mpdta):
        """Projecting on the constant alone IS the ATT.

        This is the load-bearing test for the whole projection influence
        function: with a single column of ones the weight matrix collapses
        to one row of 1/N, and both the direct and the FE-adjustment terms
        must reduce to the ATT's. Equality here is what licenses reusing
        the estimator's influence function for arbitrary weights instead
        of bolting on a separate regression sandwich.
        """
        base = sp.did_imputation(mpdta, **KEYS)
        row = (
            sp.did_imputation(mpdta, **KEYS, project=[]).diagnostics["project"].iloc[0]
        )
        assert row.term == "_cons"
        # ``abs=1e-15`` on an ATT of order 5e-2, i.e. ~2e-14 relative — the
        # same "floating-point noise" band the SE assertion below uses, and
        # the one the module docstring claims. Bitwise ``==`` was flaky: the
        # projection and the estimator reach the same number by different
        # summation orders, and a BLAS/pandas build that reassociates one of
        # them lands 1 ULP away (observed on the pandas 3.x CI leg:
        # -0.04770991827842726 vs -0.047709918278427264). Any *real* drift in
        # the shared influence function moves this far more than 1e-15.
        assert row.coef == pytest.approx(
            base.estimate, abs=1e-15
        ), "projection constant must BE the ATT"
        assert row.se == pytest.approx(
            base.se, abs=1e-15
        ), "projection SE on a constant must reduce to the ATT's analytic SE"

    def test_project_recovers_a_planted_linear_gradient(self, mpdta):
        """Correctness beyond parity: plant a known slope and recover it.

        Stata parity pins agreement with another implementation; it does
        not prove either is right. Adding a treatment effect that varies
        linearly in a covariate gives an independent check that the slope
        means what the docstring says it means.
        """
        data = mpdta.copy()
        treated = (data["first_treat"] != 0) & (data["year"] >= data["first_treat"])
        gradient = 0.25
        centred = data["lpop"] - data["lpop"].mean()
        data.loc[treated, "lemp"] = (
            data.loc[treated, "lemp"] + gradient * centred[treated]
        )

        res = sp.did_imputation(data, **KEYS, project=["lpop"])
        got = {r.term: r.coef for r in res.diagnostics["project"].itertuples()}
        baseline = sp.did_imputation(mpdta, **KEYS, project=["lpop"])
        base_slope = {
            r.term: r.coef for r in baseline.diagnostics["project"].itertuples()
        }["lpop"]
        assert got["lpop"] - base_slope == pytest.approx(gradient, abs=0.02), (
            f"planted gradient {gradient} not recovered: slope moved from "
            f"{base_slope:.4f} to {got['lpop']:.4f}"
        )


class TestProjectValidation:
    def test_project_and_hetby_are_mutually_exclusive(self, mpdta):
        with pytest.raises(ValueError, match="cannot be combined"):
            sp.did_imputation(mpdta, **KEYS, project=["lpop"], hetby="first_treat")

    def test_unknown_project_column_rejected(self, mpdta):
        with pytest.raises(ValueError, match="project column"):
            sp.did_imputation(mpdta, **KEYS, project=["nope"])

    def test_collinear_projection_rejected(self, mpdta):
        """A covariate constant among treated rows is collinear with _cons."""
        data = mpdta.copy()
        data["flat"] = 1.0
        with pytest.raises(ValueError, match="collinear"):
            sp.did_imputation(data, **KEYS, project=["flat"])

    def test_project_vars_recorded(self, mpdta):
        res = sp.did_imputation(mpdta, **KEYS, project=["lpop"])
        assert res.diagnostics["project_vars"] == ["lpop"]
