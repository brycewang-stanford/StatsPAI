"""Stata ``did_imputation`` parity for the Y(0)-model options.

Covers the three ways Stata lets you respecify the model of Y(0), which
StatsPAI previously collapsed into a single flat ``controls`` list:

===========================  ==========================================
Stata                        StatsPAI
===========================  ==========================================
``fe(i t)``  (default)       ``fe=None`` / ``fe=['unit','time']``
``fe(t)``                    ``fe=['year']``
``fe(.)``                    ``fe=[]``
``unitcontrols(year)``       ``unit_covariates=['year']``
``timecontrols(lpop)``       ``time_covariates=['lpop']``
``controls(lpop)``           ``controls=['lpop']``
===========================  ==========================================

Golden numbers from Stata 18 MP, ``did_imputation`` (2023-11-22 build) on
``mpdta``; generating do-file ``tests/stata_parity/84_bjs_fe_covariates.do``.

Tolerance
---------
1e-6 on the ATT. StatsPAI solves the untreated Y(0) fit with a sparse
``lsqr`` while Stata absorbs via ``reghdfe``, so agreement is
iterative-vs-direct and lands at ~2e-7 across every variant here.
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

ATOL = 1e-6

KEYS = dict(y="lemp", group="countyreal", time="year", first_treat="first_treat")


@pytest.fixture
def mpdta() -> pd.DataFrame:
    return pd.read_csv(_MPDTA)


@pytest.fixture
def mpdta_identified(mpdta: pd.DataFrame) -> pd.DataFrame:
    """Units with >= 2 untreated periods.

    ``unit_covariates`` gives each unit its own slope, which needs two
    untreated observations per unit. 100 of mpdta's treated observations
    sit in units with exactly one, and Stata refuses the whole estimation
    there (rc 481), so the parity comparison runs on the subset where both
    packages agree the model is identified.
    """
    untr = (mpdta["first_treat"] == 0) | (mpdta["year"] < mpdta["first_treat"])
    keep = untr.groupby(mpdta["countyreal"]).transform("sum") >= 2
    return mpdta[keep].copy()


class TestFixedEffectSpec:
    @pytest.mark.parametrize(
        "fe,expected",
        [
            (None, -0.047710015124267),  # did_imputation ... (default)
            (["countyreal", "year"], -0.047710015124267),  # fe(i t), explicit
            (["year"], 0.411346594258703),  # fe(t)
            ([], 0.358597062890112),  # fe(.)
        ],
        ids=["default", "fe_unit_time", "fe_time_only", "fe_none"],
    )
    def test_fe_variants_match_stata(self, mpdta, fe, expected):
        res = sp.did_imputation(mpdta, **KEYS, fe=fe)
        assert res.estimate == pytest.approx(expected, abs=ATOL)

    def test_explicit_two_way_fe_is_bit_identical_to_default(self, mpdta):
        """fe=['unit','time'] must not merely approximate the default path."""
        default = sp.did_imputation(mpdta, **KEYS)
        explicit = sp.did_imputation(mpdta, **KEYS, fe=["countyreal", "year"])
        assert explicit.estimate == default.estimate
        assert explicit.se == default.se

    def test_interacted_fe_spec_parses(self, mpdta):
        """``a#b`` builds the interacted cell rather than two separate FEs."""
        data = mpdta.copy()
        data["state"] = data["countyreal"] // 1000
        interacted = sp.did_imputation(data, **KEYS, fe=["countyreal", "state#year"])
        separate = sp.did_imputation(data, **KEYS, fe=["countyreal", "state", "year"])
        assert interacted.estimate != separate.estimate

    def test_fe_as_bare_string_is_rejected(self, mpdta):
        """fe='year' would silently iterate characters — reject it loudly."""
        with pytest.raises(ValueError, match="sequence of specs, not a bare string"):
            sp.did_imputation(mpdta, **KEYS, fe="year")

    def test_unknown_fe_column_rejected(self, mpdta):
        with pytest.raises(ValueError, match="not in the"):
            sp.did_imputation(mpdta, **KEYS, fe=["no_such_col"])


class TestInteractedCovariates:
    def test_timecontrols_matches_stata(self, mpdta):
        res = sp.did_imputation(mpdta, **KEYS, time_covariates=["lpop"])
        assert res.estimate == pytest.approx(-0.050627011117740, abs=ATOL)

    def test_unitcontrols_matches_stata(self, mpdta_identified):
        res = sp.did_imputation(mpdta_identified, **KEYS, unit_covariates=["year"])
        assert res.estimate == pytest.approx(-0.029441151707260, abs=ATOL)

    def test_identified_subset_default_matches_stata(self, mpdta_identified):
        """Pins the subset itself, so a fixture drift cannot fake the above."""
        res = sp.did_imputation(mpdta_identified, **KEYS)
        assert res.estimate == pytest.approx(-0.033715967954549, abs=ATOL)

    def test_unit_trends_move_the_estimate(self, mpdta_identified):
        with_trend = sp.did_imputation(
            mpdta_identified, **KEYS, unit_covariates=["year"]
        ).estimate
        without = sp.did_imputation(mpdta_identified, **KEYS).estimate
        assert abs(with_trend - without) > 1e-3, (
            "unit-specific trends should change Y(0); if they do not, the "
            "interaction columns are not entering the design"
        )

    def test_column_scaling_is_exact_not_approximate(self, mpdta_identified):
        """Equilibration must not perturb the answer it stabilizes.

        The same regressor on a different scale spans the same columns, so
        the ATT must be invariant. Before column equilibration the raw
        `year` version sat 1.6e-4 from Stata while the rescaled one sat at
        6e-8; both must now agree with each other far more tightly than
        the parity tolerance.
        """
        data = mpdta_identified.copy()
        data["yr_scaled"] = (data["year"] - data["year"].mean()) / data["year"].std()
        raw = sp.did_imputation(data, **KEYS, unit_covariates=["year"]).estimate
        scaled = sp.did_imputation(data, **KEYS, unit_covariates=["yr_scaled"]).estimate
        assert raw == pytest.approx(scaled, abs=1e-9)


class TestIdentificationGuard:
    """§7: an unidentified fit must fail loudly, not return lsqr's
    minimum-norm answer dressed up as an estimate."""

    def test_unit_covariates_without_enough_untreated_periods_raises(self, mpdta):
        """Stata errors here with rc 481; StatsPAI must not silently answer."""
        with pytest.raises(ValueError, match="at least 2 untreated observations"):
            sp.did_imputation(mpdta, **KEYS, unit_covariates=["year"])

    def test_guard_names_the_offending_units(self, mpdta):
        with pytest.raises(ValueError) as exc:
            sp.did_imputation(mpdta, **KEYS, unit_covariates=["year"])
        msg = str(exc.value)
        assert "untreated" in msg and "unit_covariates" in msg
        assert "1 untreated" in msg, "should report the actual shortfall count"

    def test_threshold_scales_with_the_number_of_slopes(self, mpdta_identified):
        """The threshold tracks the slope count, not a hard-coded 2.

        On this subset the surviving treated cohorts carry 3 and 4
        untreated periods, so two slopes (needing 3) still identify. Three
        slopes need 4 and must knock out the 3-period cohort.
        """
        data = mpdta_identified.copy()
        data["yr2"] = data["year"] ** 2.0
        data["yr3"] = data["year"] ** 3.0

        # Two slopes: still identified everywhere here.
        sp.did_imputation(data, **KEYS, unit_covariates=["year", "yr2"])

        # Three slopes: the 3-untreated-period cohort can no longer support
        # an intercept plus three slopes.
        with pytest.raises(ValueError, match="at least 4 untreated observations"):
            sp.did_imputation(data, **KEYS, unit_covariates=["year", "yr2", "yr3"])

    def test_covariate_in_both_controls_and_interacted_rejected(self, mpdta):
        with pytest.raises(ValueError, match="both `controls`"):
            sp.did_imputation(
                mpdta, **KEYS, controls=["lpop"], time_covariates=["lpop"]
            )

    def test_unknown_interacted_column_rejected(self, mpdta):
        with pytest.raises(ValueError, match="not found in data"):
            sp.did_imputation(mpdta, **KEYS, time_covariates=["nope"])
