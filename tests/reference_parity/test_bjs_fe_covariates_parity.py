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
``mpdta``; generating do-file
``tests/stata_parity/option_parity/84_bjs_fe_covariates.do``, read here from
``option_parity/results/84_bjs_fe_covariates_Stata.json``.

Tolerance
---------
Relative 1e-6 on the ATT, the default same-byte budget, for every variant
but one. Observed: 2e-14 to 5e-14 for ``fe(t)`` / ``fe(.)``, 3e-8 to 4e-7
where a unit effect is absorbed.

The first version of the fixture sat 2e-6 to 3.4e-6 away and was held to
an absolute 1e-6. That gap was Stata running in single precision:
``import delimited`` stored ``lemp`` as float and ``did_imputation``'s
generated variables followed ``set type float``. The do-file now imports
``asdouble`` under ``set type double`` with ``tol(1e-12)``.

What is left is the reference's own iteration. StatsPAI's sparse solve
agrees with a dense exact least-squares fit to 1e-11 or better on every
specification (``TestExactSolution``); Stata's alternating projections
stop 6e-8 to 1.4e-6 from it. For ``unitcontrols(year)`` the reference
does not settle below that: ``tol(1e-6)``, ``1e-10``, ``1e-12`` and
``1e-14`` return ATTs spread over 6.4e-7 relative, not monotonically. That
row is therefore held to 5e-6 and is a reference-precision disclosure, not
a same-byte match; the claim that StatsPAI is right rests on the exact
solution.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_MPDTA = (
    pathlib.Path(__file__).resolve().parents[1]
    / "orig_parity"
    / "data"
    / "02_mpdta_original.csv"
)

_STATA = json.loads(
    (
        pathlib.Path(__file__).resolve().parents[1]
        / "stata_parity"
        / "option_parity"
        / "results"
        / "84_bjs_fe_covariates_Stata.json"
    ).read_text(encoding="utf-8")
)

#: Default same-byte budget (relative).
RTOL = 1e-6
#: ``unitcontrols(year)``: the reference's iteration does not settle below
#: ~1e-6 (see the module docstring); observed gap 1.4e-6.
RTOL_UNIT_SLOPES = 5e-6

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
            (None, "default"),  # did_imputation ... (default)
            (["countyreal", "year"], "default"),  # fe(i t), explicit
            (["year"], "fe_time_only"),  # fe(t)
            ([], "fe_none"),  # fe(.)
        ],
        ids=["default", "fe_unit_time", "fe_time_only", "fe_none"],
    )
    def test_fe_variants_match_stata(self, mpdta, fe, expected):
        res = sp.did_imputation(mpdta, **KEYS, fe=fe)
        assert res.estimate == pytest.approx(_STATA[expected]["att"], rel=RTOL)

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
        assert res.estimate == pytest.approx(
            _STATA["timecontrols_lpop"]["att"], rel=RTOL
        )

    def test_controls_matches_stata(self, mpdta):
        res = sp.did_imputation(mpdta, **KEYS, controls=["lpop"])
        assert res.estimate == pytest.approx(_STATA["controls_lpop"]["att"], rel=RTOL)

    def test_unitcontrols_matches_stata(self, mpdta_identified):
        res = sp.did_imputation(mpdta_identified, **KEYS, unit_covariates=["year"])
        assert res.estimate == pytest.approx(
            _STATA["unitcontrols_year_subset"]["att"], rel=RTOL_UNIT_SLOPES
        )

    def test_identified_subset_default_matches_stata(self, mpdta_identified):
        """Pins the subset itself, so a fixture drift cannot fake the above."""
        res = sp.did_imputation(mpdta_identified, **KEYS)
        assert res.estimate == pytest.approx(
            _STATA["default_identified_subset"]["att"], rel=RTOL
        )

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


def _exact_att(frame: pd.DataFrame, *, fe: str, unit_slopes: bool = False) -> float:
    """Imputation ATT from a dense least-squares fit of Y(0).

    Dummies for the requested effects (and, optionally, a unit-specific
    slope in ``year``) are fitted on the untreated observations by
    ``numpy.linalg.lstsq`` (SVD), with no iteration and no absorption; the
    ATT is the mean of ``y - yhat0`` over the treated ones.
    """
    untreated = (
        (frame["first_treat"] == 0) | (frame["year"] < frame["first_treat"])
    ).to_numpy()
    blocks = [np.ones((len(frame), 1))]
    unit = pd.get_dummies(frame["countyreal"], dtype=float).to_numpy()
    if "unit" in fe:
        blocks.append(unit)
    if "time" in fe:
        blocks.append(pd.get_dummies(frame["year"], dtype=float).to_numpy())
    if unit_slopes:
        centred = (frame["year"] - frame["year"].mean()).to_numpy()
        blocks.append(unit * centred[:, None])
    design = np.hstack(blocks)
    y = frame["lemp"].to_numpy()
    coef, *_ = np.linalg.lstsq(design[untreated], y[untreated], rcond=None)
    return float((y - design @ coef)[~untreated].mean())


class TestExactSolution:
    """The independent evidence behind the reference-precision disclosure.

    Where StatsPAI and Stata differ by more than rounding, one of them is
    off the least-squares answer. These pin StatsPAI to it, so the residual
    gap in the Stata rows is the reference's.
    """

    @pytest.mark.parametrize(
        "fe,spec",
        [(None, "unit+time"), (["year"], "time"), ([], "")],
        ids=["two_way", "time_only", "none"],
    )
    def test_fe_variants_equal_the_dense_fit(self, mpdta, fe, spec):
        res = sp.did_imputation(mpdta, **KEYS, fe=fe)
        assert res.estimate == pytest.approx(_exact_att(mpdta, fe=spec), rel=1e-9)

    def test_identified_subset_equals_the_dense_fit(self, mpdta_identified):
        res = sp.did_imputation(mpdta_identified, **KEYS)
        assert res.estimate == pytest.approx(
            _exact_att(mpdta_identified, fe="unit+time"), rel=1e-9
        )

    def test_unit_slopes_equal_the_dense_fit(self, mpdta_identified):
        """Observed 7.5e-12; Stata sits 1.4e-6 from the same number."""
        res = sp.did_imputation(mpdta_identified, **KEYS, unit_covariates=["year"])
        exact = _exact_att(mpdta_identified, fe="unit+time", unit_slopes=True)
        assert res.estimate == pytest.approx(exact, rel=1e-9)
        stata = _STATA["unitcontrols_year_subset"]["att"]
        assert abs(res.estimate - exact) < 1e-3 * abs(stata - exact)


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
