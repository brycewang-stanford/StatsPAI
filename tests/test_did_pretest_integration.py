"""Integrated pre-trend tests on the staggered DiD estimators.

``sp.callaway_santanna`` always computed a joint pre-trend test but gave
no way to scope it; ``sp.sun_abraham`` had **no** pre-trend test at all,
so callers had to read the event-study table by eye — the standard way to
conclude "parallel trends hold" from leads that are individually
insignificant but jointly not.

Both now take ``pretest=`` / ``pretest_periods=``.

There is no Stata counterpart to pin here: ``eventstudyinteract`` reports
no joint test, and ``csdid``'s aggregation differs. These are therefore
*analytic* tests — the statistic's own properties (degrees of freedom,
covariance handling, invariance, power) rather than cross-package
agreement.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_MPDTA = (
    pathlib.Path(__file__).resolve().parent
    / "orig_parity"
    / "data"
    / "02_mpdta_original.csv"
)

CS = dict(y="lemp", g="first_treat", t="year", i="countyreal")


@pytest.fixture(scope="module")
def mpdta() -> pd.DataFrame:
    return pd.read_csv(_MPDTA)


def _trending_panel(slope: float, n_units: int = 200, seed: int = 4) -> pd.DataFrame:
    """Panel whose treated cohorts drift before treatment.

    ``slope=0`` satisfies parallel trends exactly; larger values violate
    it more strongly, which is what gives the power check its teeth.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = [0, 5, 7][u % 3]
        drift = slope if g > 0 else 0.0
        for t in range(1, 11):
            d = 1 if (g > 0 and t >= g) else 0
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "gvar": g,
                    "y": 0.5 * d + drift * t + 0.02 * u + rng.normal(0, 0.3),
                }
            )
    return pd.DataFrame(rows)


class TestSunAbrahamPretrend:
    def test_test_is_reported_by_default(self, mpdta):
        res = sp.sun_abraham(mpdta, **CS)
        t = res.diagnostics["pretrend_test"]
        assert t is not None
        assert set(t) >= {"statistic", "df", "pvalue", "relative_times"}
        assert t["df"] == len(t["relative_times"])

    def test_none_disables(self, mpdta):
        assert (
            sp.sun_abraham(mpdta, **CS, pretest="none").diagnostics["pretrend_test"]
            is None
        )

    def test_covers_every_estimated_lead_by_default(self, mpdta):
        res = sp.sun_abraham(mpdta, **CS)
        leads = sorted(int(e) for e in res.detail["relative_time"] if e < 0)
        assert res.diagnostics["pretrend_test"]["relative_times"] == leads

    @pytest.mark.parametrize("k,expected_n", [(1, 1), (2, 2), (3, 3), (99, 3)])
    def test_pretest_periods_counts_estimated_leads(self, mpdta, k, expected_n):
        """k counts leads that exist, and saturates rather than erroring.

        mpdta's estimated leads are -4, -3, -2 (ℓ = -1 is the omitted
        reference), so a literal ``ℓ >= -k`` cutoff would return one fewer
        than asked for at every k — and nothing at all for k=1.
        """
        t = sp.sun_abraham(mpdta, **CS, pretest_periods=k).diagnostics["pretrend_test"]
        assert t["df"] == expected_n
        assert len(t["relative_times"]) == expected_n
        assert t["relative_times"] == sorted(t["relative_times"])[-expected_n:]

    def test_selects_the_leads_nearest_treatment(self, mpdta):
        t = sp.sun_abraham(mpdta, **CS, pretest_periods=2).diagnostics["pretrend_test"]
        assert t["relative_times"] == [-3, -2], "must keep the NEAREST leads"

    def test_uses_the_joint_covariance_not_the_diagonal(self, mpdta):
        """A diagonal-covariance Wald would be a different number.

        Summing squared t-statistics is what you get by pretending the
        leads are independent. The real statistic must differ from it —
        the leads share control units and are correlated by construction.
        """
        res = sp.sun_abraham(mpdta, **CS)
        leads = res.detail[res.detail["relative_time"] < 0]
        naive = float(((leads["att"] / leads["se"]) ** 2).sum())
        actual = res.diagnostics["pretrend_test"]["statistic"]
        assert abs(actual - naive) > 1e-6, (
            f"statistic {actual:.6f} equals the independence-assuming sum "
            f"{naive:.6f}; the off-diagonal covariance is being ignored"
        )

    def test_flat_trends_are_not_rejected(self):
        res = sp.sun_abraham(_trending_panel(0.0), y="y", g="gvar", t="time", i="unit")
        assert res.diagnostics["pretrend_test"]["pvalue"] > 0.05

    def test_strong_differential_trend_is_rejected(self):
        """Power: a pre-trend the estimator cannot see is worthless."""
        res = sp.sun_abraham(_trending_panel(0.30), y="y", g="gvar", t="time", i="unit")
        assert res.diagnostics["pretrend_test"]["pvalue"] < 0.01

    def test_statistic_is_nonnegative_and_df_positive(self, mpdta):
        t = sp.sun_abraham(mpdta, **CS).diagnostics["pretrend_test"]
        assert t["statistic"] >= 0
        assert t["df"] >= 1
        assert 0.0 <= t["pvalue"] <= 1.0

    @pytest.mark.parametrize("bad", ["yes", "individual", 1])
    def test_invalid_pretest_rejected(self, mpdta, bad):
        with pytest.raises(ValueError, match="pretest must be"):
            sp.sun_abraham(mpdta, **CS, pretest=bad)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True])
    def test_invalid_pretest_periods_rejected(self, mpdta, bad):
        with pytest.raises(ValueError, match="pretest_periods"):
            sp.sun_abraham(mpdta, **CS, pretest_periods=bad)


class TestCallawaySantannaPretrend:
    def test_default_is_unchanged(self, mpdta):
        """The historical test must keep its historical value.

        pretest= is additive: anyone who never passes it must get exactly
        the number they got before.
        """
        t = sp.callaway_santanna(mpdta, **CS).diagnostics["pretrend_test"]
        assert t["df"] == 5
        assert t["statistic"] == pytest.approx(7.7912309648774976, rel=1e-9)
        assert t["pvalue"] == pytest.approx(0.1740323045946247, rel=1e-9)

    def test_none_disables(self, mpdta):
        assert (
            sp.callaway_santanna(mpdta, **CS, pretest="none").diagnostics[
                "pretrend_test"
            ]
            is None
        )

    def test_restricting_periods_reduces_df(self, mpdta):
        """CS's df counts (g,t) cells, not event times."""
        full = sp.callaway_santanna(mpdta, **CS).diagnostics["pretrend_test"]
        near = sp.callaway_santanna(mpdta, **CS, pretest_periods=1).diagnostics[
            "pretrend_test"
        ]
        assert near["df"] < full["df"]
        assert near["df"] >= 1

    def test_repeated_cross_sections_branch_honours_the_option(self, mpdta):
        """panel=False routes through a separate code path — thread it too."""
        assert (
            sp.callaway_santanna(mpdta, **CS, panel=False, pretest="none").diagnostics[
                "pretrend_test"
            ]
            is None
        )
        assert (
            sp.callaway_santanna(mpdta, **CS, panel=False).diagnostics["pretrend_test"]
            is not None
        )

    def test_flat_trends_are_not_rejected(self):
        res = sp.callaway_santanna(
            _trending_panel(0.0), y="y", g="gvar", t="time", i="unit"
        )
        assert res.diagnostics["pretrend_test"]["pvalue"] > 0.05

    def test_strong_differential_trend_is_rejected(self):
        res = sp.callaway_santanna(
            _trending_panel(0.30), y="y", g="gvar", t="time", i="unit"
        )
        assert res.diagnostics["pretrend_test"]["pvalue"] < 0.01


class TestBothEstimatorsAgreeQualitatively:
    """Two different estimators should not disagree about a loud pre-trend."""

    @pytest.mark.parametrize("slope,rejects", [(0.0, False), (0.30, True)])
    def test_same_verdict_on_the_same_panel(self, slope, rejects):
        panel = _trending_panel(slope)
        keys = dict(y="y", g="gvar", t="time", i="unit")
        cs_p = sp.callaway_santanna(panel, **keys).diagnostics["pretrend_test"][
            "pvalue"
        ]
        sa_p = sp.sun_abraham(panel, **keys).diagnostics["pretrend_test"]["pvalue"]
        assert (cs_p < 0.01) is rejects
        assert (sa_p < 0.01) is rejects
