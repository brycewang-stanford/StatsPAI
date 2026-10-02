"""Stata ``csdid`` parity for the CS convention switches.

Covers the three option axes that Stata exposes and StatsPAI gained in the
DiD option-depth campaign:

* ``notyet_cutoff`` — csdid's ``asinr`` (``notyet_cutoff='asinr'``) versus
  csdid's own default (``'cohort'``) for pre-treatment ATT(g,t).  StatsPAI's
  default ``'period'`` is R ``did``'s rule, ``G > max(t, base) +
  anticipation``; under the universal base used here it coincides with
  csdid's default on every pre-treatment cell of ``mpdta``, and R ``did``
  2.3.0 reports those same numbers (the full R pin is in
  ``test_cs_notyet_cutoff_parity.py``).  Until 1.29 the docs described
  ``asinr`` as the R convention; that holds only under a varying base.
* ``estimator='ipw'``/``'stdipw'`` versus ``'ipw_abadie'`` — csdid's
  ``method(stdipw)`` and ``method(ipw)`` respectively.

The golden numbers were produced by Stata 18 MP with ``csdid`` v1.81 on
``mpdta``; the generating do-file is
``tests/stata_parity/option_parity/82_csdid_conventions.do`` and the numbers
are read from ``option_parity/results/82_csdid_conventions_Stata.json``.

Tolerance: relative 1e-9; the observed worst case is 4.9e-13. The first
version of the fixture sat 5e-8 away (2e-5 relative on the small
pre-treatment cells) and the gap was put down to the propensity logit's
optimizer. It was Stata running in single precision: ``import delimited``
stored ``lemp`` / ``lpop`` as float. The do-file now imports ``asdouble``
under ``set type double``.
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

# Stata is not required at test time: the committed fixture is read.
_STATA = json.loads(
    (
        pathlib.Path(__file__).resolve().parents[1]
        / "stata_parity"
        / "option_parity"
        / "results"
        / "82_csdid_conventions_Stata.json"
    ).read_text(encoding="utf-8")
)

# Observed worst case 4.9e-13; a convention regression moves a cell by
# 1e-5 or more.
RTOL = 1e-9


def _cells(block: str, keys: list) -> dict:
    """Attach (g, t) to csdid's e(b) columns, which come in this order."""
    values = list(_STATA[block].values())
    assert len(values) == len(keys) == 12
    return dict(zip(keys, values))


def _mpdta() -> pd.DataFrame:
    return pd.read_csv(_MPDTA)


def _atts(**kwargs) -> dict:
    res = sp.callaway_santanna(
        _mpdta(),
        y="lemp",
        g="first_treat",
        t="year",
        i="countyreal",
        base_period="universal",
        **kwargs,
    )
    det = res.detail
    return {(int(r.group), int(r.time)): float(r.att) for r in det.itertuples()}


# ----------------------------------------------------------------------
# csdid ... notyet long2 asinr method(reg)      [R convention = default]
# ----------------------------------------------------------------------
STATA_ASINR = _cells(
    "notyet_asinr_reg",
    [
        (2004, 2004),
        (2004, 2005),
        (2004, 2006),
        (2004, 2007),
        (2006, 2003),
        (2006, 2004),
        (2006, 2006),
        (2006, 2007),
        (2007, 2003),
        (2007, 2004),
        (2007, 2005),
        (2007, 2007),
    ],
)

# ----------------------------------------------------------------------
# csdid ... notyet long2 method(reg)            [csdid's own default]
# ----------------------------------------------------------------------
STATA_CSDID_DEFAULT = _cells(
    "notyet_csdid_default_reg",
    [
        (2004, 2004),
        (2004, 2005),
        (2004, 2006),
        (2004, 2007),
        (2006, 2003),
        (2006, 2004),
        (2006, 2006),
        (2006, 2007),
        (2007, 2003),
        (2007, 2004),
        (2007, 2005),
        (2007, 2007),
    ],
)

# ----------------------------------------------------------------------
# csdid lemp lpop ... long2 method(stdipw) / method(ipw)
# ----------------------------------------------------------------------
STATA_STDIPW = _cells(
    "stdipw_lpop",
    [
        (2004, 2004),
        (2004, 2005),
        (2004, 2006),
        (2004, 2007),
        (2006, 2003),
        (2006, 2004),
        (2006, 2006),
        (2006, 2007),
        (2007, 2003),
        (2007, 2004),
        (2007, 2005),
        (2007, 2007),
    ],
)

STATA_IPW_ABADIE = _cells(
    "ipw_abadie_lpop",
    [
        (2004, 2004),
        (2004, 2005),
        (2004, 2006),
        (2004, 2007),
        (2006, 2003),
        (2006, 2004),
        (2006, 2006),
        (2006, 2007),
        (2007, 2003),
        (2007, 2004),
        (2007, 2005),
        (2007, 2007),
    ],
)


def _assert_matches(got: dict, want: dict, label: str) -> None:
    assert set(got) == set(want), f"{label}: (g,t) cell set differs"
    for key, expected in want.items():
        assert got[key] == pytest.approx(expected, rel=RTOL), (
            f"{label}: ATT{key} = {got[key]:.12f}, Stata {expected:.12f} "
            f"(diff {abs(got[key] - expected):.2e})"
        )


class TestNotyetCutoff:
    """csdid ``asinr`` vs csdid default: notyet_cutoff='asinr'|'cohort'."""

    def test_asinr_cutoff_matches_stata_asinr(self):
        got = _atts(
            estimator="reg", control_group="notyettreated", notyet_cutoff="asinr"
        )
        _assert_matches(got, STATA_ASINR, "notyet_cutoff='asinr'")

    def test_default_period_cutoff_is_r_did_rule(self):
        """R did's rule equals csdid's default on mpdta's universal-base cells.

        R ``did::att_gt(control_group='notyettreated', base_period='universal',
        est_method='reg')`` returns ATT(2007, 2004) = 0.033813 and ATT(2006,
        2003) = 0.004502 on this file -- the csdid-default numbers, not the
        asinr ones (0.032971 / 0.001080).
        """
        got = _atts(estimator="reg", control_group="notyettreated")
        _assert_matches(got, STATA_CSDID_DEFAULT, "notyet_cutoff='period' (R did)")

    def test_cohort_cutoff_matches_stata_csdid_default(self):
        got = _atts(
            estimator="reg",
            control_group="notyettreated",
            notyet_cutoff="cohort",
        )
        _assert_matches(got, STATA_CSDID_DEFAULT, "notyet_cutoff='cohort'")

    def test_cutoff_only_moves_pre_treatment_cells(self):
        """The two conventions must agree wherever t >= g.

        This is the property that broke when the cohort cutoff was first
        applied to every cell: post-treatment ATT(2006,2007) moved from
        -0.0412 to -0.0242. Pin it directly so the scoping cannot regress.
        """
        period = _atts(
            estimator="reg", control_group="notyettreated", notyet_cutoff="asinr"
        )
        cohort = _atts(
            estimator="reg",
            control_group="notyettreated",
            notyet_cutoff="cohort",
        )
        post = [(g, t) for (g, t) in period if t >= g]
        assert post, "fixture should contain post-treatment cells"
        for key in post:
            assert cohort[key] == pytest.approx(
                period[key], abs=1e-12
            ), f"post-treatment ATT{key} must not depend on notyet_cutoff"
        pre = [(g, t) for (g, t) in period if t < g]
        assert any(
            abs(cohort[k] - period[k]) > 1e-6 for k in pre
        ), "the two conventions should differ on at least one pre-treatment cell"


class TestIpwVariants:
    """StatsPAI 'ipw' is Stata's stdipw; 'ipw_abadie' is Stata's ipw."""

    def test_ipw_matches_stata_stdipw(self):
        got = _atts(estimator="ipw", x=["lpop"])
        _assert_matches(got, STATA_STDIPW, "estimator='ipw'")

    def test_stdipw_alias_is_bit_identical_to_ipw(self):
        a = _atts(estimator="ipw", x=["lpop"])
        b = _atts(estimator="stdipw", x=["lpop"])
        for key in a:
            assert a[key] == b[key], f"'stdipw' must alias 'ipw' exactly at {key}"

    def test_ipw_abadie_matches_stata_ipw(self):
        got = _atts(estimator="ipw_abadie", x=["lpop"])
        _assert_matches(got, STATA_IPW_ABADIE, "estimator='ipw_abadie'")

    def test_abadie_and_stabilized_are_genuinely_different(self):
        """Guard against 'ipw_abadie' silently collapsing onto 'ipw'.

        Without covariates the propensity score is constant and the two
        coincide analytically, so the separation must be checked with
        covariates in play.
        """
        stab = _atts(estimator="ipw", x=["lpop"])
        abadie = _atts(estimator="ipw_abadie", x=["lpop"])
        gaps = [abs(stab[k] - abadie[k]) for k in stab]
        assert max(gaps) > 1e-5, (
            "Abadie and stabilized IPW should differ materially with "
            f"covariates; max gap was {max(gaps):.2e}"
        )

    def test_no_covariates_collapses_the_two_ipw_variants(self):
        """Boundary: constant propensity => identical weights => identical ATT."""
        stab = _atts(estimator="ipw")
        abadie = _atts(estimator="ipw_abadie")
        for key in stab:
            assert abadie[key] == pytest.approx(
                stab[key], abs=1e-10
            ), f"with no covariates the IPW variants must coincide at {key}"


class TestPscoreTrim:
    """Control-side propensity trimming, matching DRDID's trim.level."""

    def test_default_trim_is_inert_on_mpdta(self):
        """mpdta has no control with p(X) >= 0.995, so 0.995 and 1.0 agree.

        This is what licenses the golden numbers above: they were produced
        by Stata, which trims at 0.995, and match StatsPAI's untrimmed
        history bit for bit.
        """
        trimmed = _atts(estimator="dr", x=["lpop"])
        untrimmed = _atts(estimator="dr", x=["lpop"], pscore_trim=1.0)
        for key in trimmed:
            assert trimmed[key] == pytest.approx(untrimmed[key], abs=1e-12)

    def test_trim_binds_and_warns_when_overlap_is_poor(self):
        """An aggressive cutoff must bite, be counted, and say so."""
        data = _mpdta()
        with pytest.warns(UserWarning, match="propensity trimming removed"):
            res = sp.callaway_santanna(
                data,
                y="lemp",
                g="first_treat",
                t="year",
                i="countyreal",
                x=["lpop"],
                estimator="dr",
                pscore_trim=0.30,
            )
        assert res.diagnostics["n_pscore_trimmed"] > 0
        assert res.diagnostics["pscore_trim"] == 0.30

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, np.nan, "0.995", True])
    def test_invalid_trim_rejected(self, bad):
        with pytest.raises(Exception):
            sp.callaway_santanna(
                _mpdta(),
                y="lemp",
                g="first_treat",
                t="year",
                i="countyreal",
                pscore_trim=bad,
            )

    def test_inert_cutoff_on_nevertreated_warns(self):
        with pytest.warns(UserWarning, match="only affects the 'notyettreated'"):
            sp.callaway_santanna(
                _mpdta(),
                y="lemp",
                g="first_treat",
                t="year",
                i="countyreal",
                control_group="nevertreated",
                notyet_cutoff="cohort",
            )
