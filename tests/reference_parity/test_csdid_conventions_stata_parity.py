"""Stata ``csdid`` parity for the CS convention switches.

Covers the three option axes that Stata exposes and StatsPAI gained in the
DiD option-depth campaign:

* ``notyet_cutoff`` — csdid's ``asinr`` (R convention, StatsPAI default)
  versus csdid's own default for pre-treatment ATT(g,t).
* ``estimator='ipw'``/``'stdipw'`` versus ``'ipw_abadie'`` — csdid's
  ``method(stdipw)`` and ``method(ipw)`` respectively.

The golden numbers were produced by Stata 18 MP with ``csdid`` v1.81 on
``mpdta``; the generating do-file is ``tests/stata_parity/82_csdid_conventions.do``.
Point estimates are pinned; the residual gap is propensity-logit optimizer
tolerance (statsmodels Newton vs Stata's ML), which lands at ~5e-8.
"""

from __future__ import annotations

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

# Stata is not required at test time — these are committed golden values.
# atol is set one order above the observed 5.2e-8 worst case so that a
# genuine convention regression (>= 1e-5, as seen when the cutoff was
# wrongly applied to post-treatment cells) fails loudly while optimizer
# noise does not.
ATOL = 5e-7


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
STATA_ASINR = {
    (2004, 2004): -0.019372312476,
    (2004, 2005): -0.078319045405,
    (2004, 2006): -0.136274308508,
    (2004, 2007): -0.100811371526,
    (2006, 2003): 0.001080322525,
    (2006, 2004): 0.001939205961,
    (2006, 2006): 0.004660857808,
    (2006, 2007): -0.041224479598,
    (2007, 2003): -0.004222612298,
    (2007, 2004): 0.032971100525,
    (2007, 2005): 0.030560489610,
    (2007, 2007): -0.026054398766,
}

# ----------------------------------------------------------------------
# csdid ... notyet long2 method(reg)            [csdid's own default]
# ----------------------------------------------------------------------
STATA_CSDID_DEFAULT = {
    (2004, 2004): -0.019372312476,
    (2004, 2005): -0.078319045405,
    (2004, 2006): -0.136274308508,
    (2004, 2007): -0.100811371526,
    (2006, 2003): 0.004501809315,
    (2006, 2004): 0.001939205961,
    (2006, 2006): 0.004660857808,
    (2006, 2007): -0.041224479598,
    (2007, 2003): 0.003306351642,
    (2007, 2004): 0.033812979532,
    (2007, 2005): 0.031087093360,
    (2007, 2007): -0.026054398766,
}

# ----------------------------------------------------------------------
# csdid lemp lpop ... long2 method(stdipw) / method(ipw)
# ----------------------------------------------------------------------
STATA_STDIPW = {
    (2004, 2004): -0.014548390416,
    (2004, 2005): -0.076449812427,
    (2004, 2006): -0.140464559636,
    (2004, 2007): -0.106932567121,
    (2006, 2003): 0.007265814814,
    (2006, 2004): 0.006397198285,
    (2006, 2006): 0.001208037786,
    (2006, 2007): -0.041308248592,
    (2007, 2003): 0.006445096473,
    (2007, 2004): 0.033001174450,
    (2007, 2005): 0.028340276029,
    (2007, 2007): -0.028894759817,
}

STATA_IPW_ABADIE = {
    (2004, 2004): -0.014541146050,
    (2004, 2005): -0.076444396927,
    (2004, 2006): -0.140463010488,
    (2004, 2007): -0.106934128119,
    (2006, 2003): 0.007102507260,
    (2006, 2004): 0.006466480172,
    (2006, 2006): 0.001088945174,
    (2006, 2007): -0.041545749606,
    (2007, 2003): 0.006434015814,
    (2007, 2004): 0.033041718038,
    (2007, 2005): 0.028367848826,
    (2007, 2007): -0.028916821396,
}


def _assert_matches(got: dict, want: dict, label: str) -> None:
    assert set(got) == set(want), f"{label}: (g,t) cell set differs"
    for key, expected in want.items():
        assert got[key] == pytest.approx(expected, abs=ATOL), (
            f"{label}: ATT{key} = {got[key]:.12f}, Stata {expected:.12f} "
            f"(diff {abs(got[key] - expected):.2e})"
        )


class TestNotyetCutoff:
    """csdid ``asinr`` vs csdid default, i.e. notyet_cutoff='period'|'cohort'."""

    def test_period_cutoff_matches_stata_asinr(self):
        got = _atts(estimator="reg", control_group="notyettreated")
        _assert_matches(got, STATA_ASINR, "notyet_cutoff='period'")

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
        period = _atts(estimator="reg", control_group="notyettreated")
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
