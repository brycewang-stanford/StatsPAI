"""Regression tests for four defects found while closing the core coverage gap.

Each one returned a wrong number or an unrelated crash on a legitimate or
degenerate input; none was reachable from the existing suite.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.rd._rdplot_core import rdplot_numbers


def _staggered_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(90):
        g = [2003, 2005, 0][i % 3]
        for t in range(2000, 2008):
            d = int(g > 0 and t >= g)
            rows.append((i, t, g, rng.normal() + 0.3 * (t - 2000) + 2.0 * d))
    return pd.DataFrame(rows, columns=["id", "t", "g", "y"])


class TestFectForceNone:
    def test_fe_without_fixed_effects_is_a_difference_in_means(self):
        # With no unit or period effects and no factors the counterfactual
        # for every treated cell is the mean of the untreated cells, so the
        # ATT is the difference of the two means. R fect 2.4.1 returns the
        # same number on this panel (2.387824).
        rng = np.random.default_rng(0)
        rows = []
        for i in range(30):
            for t in range(10):
                d = int(i < 8 and t >= 6)
                rows.append((i, t, d, 5.0 + rng.normal(0, 1) + 2.0 * d))
        df = pd.DataFrame(rows, columns=["id", "time", "d", "y"])
        expected = df.loc[df.d == 1, "y"].mean() - df.loc[df.d == 0, "y"].mean()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sp.fect(
                df, y="y", treat="d", unit="id", time="time", method="fe", force="none"
            )

        # Closed form; the only slack is floating-point summation order.
        assert fit.estimate == pytest.approx(expected, abs=1e-10)
        assert fit.estimate == pytest.approx(2.387824, abs=5e-7)


class TestCalendarCohortsWithMissingPeriods:
    @pytest.mark.parametrize(
        "estimator", [sp.sun_abraham, sp.callaway_santanna], ids=["sa", "cs"]
    )
    def test_period_cohorts_with_nat_match_integer_coding(self, estimator):
        df = _staggered_panel()
        periods = df.copy()
        periods["t"] = pd.PeriodIndex(df.t.astype(str), freq="Y")
        periods["g"] = pd.PeriodIndex(
            [str(v) if v > 0 else "NaT" for v in df.g], freq="Y"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            by_integer = estimator(df, y="y", g="g", t="t", i="id")
            by_period = estimator(periods, y="y", g="g", t="t", i="id")

        # Same data under another encoding: only the labels differ.
        assert by_period.estimate == pytest.approx(by_integer.estimate, abs=1e-10)
        assert by_period.se == pytest.approx(by_integer.se, abs=1e-10)

    def test_never_treated_period_cohorts_are_coded_zero(self):
        from statspai.did._core import index_calendar_time

        frame = pd.DataFrame(
            {
                "t": pd.PeriodIndex(["2001", "2002", "2003", "2004"] * 4, freq="Y"),
                "g": pd.PeriodIndex(
                    ["2002"] * 4 + ["2004"] * 4 + ["NaT"] * 4 + ["2002"] * 4,
                    freq="Y",
                ),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, _ = index_calendar_time(frame, "t", "g", function="test")

        # 2002 is the second observed period and 2004 the fourth. The unit
        # listed after the never-treated one used to be given position 2 or 1
        # depending on where the search for the NaT key had stopped.
        assert out["g"].tolist() == [2] * 4 + [4] * 4 + [0] * 4 + [2] * 4


class TestRdrobustSupport:
    def test_one_support_point_a_side_is_refused(self):
        rng = np.random.default_rng(1)
        x = np.repeat([-1.0, 1.0], 100)
        y = 1.0 * (x >= 0) + 0.2 * x + rng.normal(0, 0.3, 200)
        df = pd.DataFrame({"y": y, "x": x})

        with pytest.raises(DataInsufficient, match="too few distinct values") as err:
            sp.rdrobust(df, y="y", x="x", warn_mass_points=False)

        assert err.value.diagnostics["support_left"] == 1
        assert err.value.diagnostics["minimum_support_per_side"] == 2

    def test_order_two_needs_three_support_points(self):
        rng = np.random.default_rng(2)
        x = np.repeat([-2.0, -1.0, 1.0, 2.0], 100)
        y = 1.0 * (x >= 0) + rng.normal(0, 0.3, 400)
        df = pd.DataFrame({"y": y, "x": x})

        with pytest.raises(DataInsufficient, match="p=2"):
            sp.rdrobust(df, y="y", x="x", p=2, warn_mass_points=False)


class TestRdplotConstantSide:
    @pytest.mark.parametrize(
        "kwargs", [{}, {"nbins": (6, 6)}, {"binselect": "qs"}], ids=str
    )
    def test_constant_outcome_on_one_side_gets_one_bin(self, kwargs):
        rng = np.random.default_rng(3)
        x = rng.uniform(-1, 1, 300)
        y = np.where(x < 0, 2.0, 1 + x + rng.normal(0, 0.2, 300))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = rdplot_numbers(y, x, **kwargs)

        assert out["J"][0] == 1
        assert out["J_IMSE"][0] == 1 and out["J_MV"][0] == 1
        assert out["J"][1] > 1
        # The single left bin holds every left observation, all equal to 2.
        assert out["vars_bins"]["rdplot_mean_y"][0] == pytest.approx(2.0, abs=1e-12)


class TestSdidSeMethodThroughDispatcher:
    KW = dict(
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
    )

    def test_se_method_is_accepted_and_does_not_move_the_estimate(self):
        df = sp.datasets.california_prop99()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            default = sp.synth(df, method="sdid", **self.KW)
            named = sp.synth(df, method="sdid", se_method="placebo", **self.KW)

        # The point estimate does not depend on the inference method.
        assert named.estimate == pytest.approx(default.estimate, abs=1e-10)
        assert np.isfinite(named.se) and named.se > 0

    def test_conflicting_spellings_are_refused(self):
        df = sp.datasets.california_prop99()
        with pytest.raises(MethodIncompatibility, match="disagree"):
            sp.synth(
                df,
                method="sdid",
                se_method="placebo",
                inference="bootstrap",
                **self.KW,
            )
