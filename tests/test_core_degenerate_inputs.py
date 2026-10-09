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


def _two_period_weighted_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=n)
    d = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * x))).astype(int)
    het = rng.uniform(size=n) < 0.5
    tau = np.where(het, 1.0, 5.0)
    y0 = x + rng.normal(size=n)
    y1 = y0 + 0.5 + 0.3 * x + tau * d + rng.normal(size=n)
    w = np.where(het, 1.0, 9.0) * rng.uniform(0.5, 1.5, n)
    pre = pd.DataFrame({"id": np.arange(n), "t": 0, "y": y0, "d": d, "x": x, "w": w})
    return pd.concat([pre, pre.assign(t=1, y=y1)], ignore_index=True)


class TestDrdidPanelWeights:
    # R DRDID 1.3.0 on this panel with panel = TRUE, weightsname = "w":
    # drdid(estMethod = "imp"), drdid(estMethod = "trad"), ipwdid, ordid.
    DRDID = {
        ("imp", "dr"): (4.673879248465, 0.120516615690),
        ("trad", "dr"): (4.672770516498, 0.120707052519),
        ("trad", "ipw"): (4.677827031718, 0.119758102004),
        ("trad", "reg"): (4.663051889258, 0.121180389065),
    }

    @pytest.mark.parametrize("key", list(DRDID), ids=lambda k: "-".join(k))
    def test_weighted_panel_matches_drdid(self, key):
        method, est_method = key
        fit = sp.drdid(
            _two_period_weighted_panel(),
            y="y",
            group="d",
            time="t",
            covariates=["x"],
            id="id",
            weights="w",
            method=method,
            est_method=est_method,
        )
        att, se = self.DRDID[key]
        # Reference printed to 12 decimals. The improved estimator's two
        # nuisance fits are iterative on both sides, hence 1e-7 on its se.
        assert fit.estimate == pytest.approx(att, abs=1e-9)
        assert fit.se == pytest.approx(se, abs=1e-7)

    def test_weights_move_the_estimate_toward_the_heavier_units(self):
        # Half the units have effect 1 and weight about 1, half have effect 5
        # and weight about 9: the weighted ATT is near 4.6, the unweighted
        # near 3. The panel path used to return the unweighted number.
        df = _two_period_weighted_panel()
        kw = dict(y="y", group="d", time="t", covariates=["x"], id="id")
        plain = sp.drdid(df, **kw)
        weighted = sp.drdid(df, weights="w", **kw)
        assert plain.estimate == pytest.approx(3.1, abs=0.3)
        assert weighted.estimate == pytest.approx(4.6, abs=0.3)

    def test_negative_weights_are_refused(self):
        df = _two_period_weighted_panel()
        df.loc[df["id"] == 0, "w"] = -1.0
        with pytest.raises(MethodIncompatibility, match="non-negative"):
            sp.drdid(df, y="y", group="d", time="t", id="id", weights="w")


class TestCallawaySantannaDegenerateClusters:
    @pytest.fixture(scope="class")
    def panel(self):
        df = _staggered_panel()
        df["period_cluster"] = df["t"]
        df["region"] = df["id"] % 12
        return df

    def test_time_varying_cluster_is_refused_with_the_unbalanced_flag(self, panel):
        # The flag used to switch the check off even on a balanced panel,
        # where the ordinary panel estimator runs; the result was se = 2e-16.
        with pytest.raises(MethodIncompatibility, match="time-varying within unit"):
            sp.callaway_santanna(
                panel,
                y="y",
                g="g",
                t="t",
                i="id",
                allow_unbalanced_panel=True,
                clustervars=["period_cluster"],
            )

    def test_cluster_made_of_whole_cells_is_refused_in_cross_sections(self, panel):
        with pytest.raises(
            MethodIncompatibility, match="constant within every cohort x period cell"
        ):
            sp.callaway_santanna(
                panel,
                y="y",
                g="g",
                t="t",
                i="id",
                panel=False,
                clustervars=["period_cluster"],
            )

    def test_a_cluster_that_cuts_across_cells_still_works(self, panel):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sp.callaway_santanna(
                panel, y="y", g="g", t="t", i="id", panel=False, clustervars=["region"]
            )
        assert fit.se > 0.01


class TestDidMultiplegtNotEstimable:
    def test_placebo_and_dynamic_beyond_the_panel_are_missing(self):
        rng = np.random.default_rng(0)
        rows = []
        for unit in range(200):
            for period in (1, 2):
                d = int(unit < 80 and period == 2)
                rows.append((unit, period, d, rng.normal() + 1.5 * d))
        df = pd.DataFrame(rows, columns=["g", "t", "d", "y"])

        with pytest.warns(UserWarning, match="placebo -1, dynamic 1"):
            fit = sp.did_multiplegt(
                df,
                y="y",
                group="g",
                time="t",
                treatment="d",
                placebo=1,
                dynamic=1,
                n_boot=20,
                seed=1,
            )

        es = fit.model_info["event_study"].set_index("relative_time")
        # Two periods: no period before the switch, none after it.
        assert es.loc[[-1, 1], ["att", "se", "pvalue"]].isna().all().all()
        # The instantaneous effect is estimable and unchanged by the request.
        assert es.loc[0, "att"] == pytest.approx(fit.estimate, abs=1e-12)
        assert fit.model_info["joint_placebo_test"] is None
        assert fit.model_info["avg_cumulative_effect"]["n_horizons"] == 1


class TestRelativeMagnitudeSetFarFromZero:
    def test_set_is_found_when_the_estimate_is_outside_the_default_grid(self):
        from statspai.did._arp import rm_confidence_set

        # t = 5 / sqrt(0.05) = 22.4, beyond the +/-20 sd default grid.
        sigma = np.diag([1e-4, 1e-4, 0.05])
        beta = np.array([0.0, 0.0, 5.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lo, hi, grid, _ = rm_confidence_set(
                beta, sigma, 2, 1, 0.0, method="C-LF", grid_points=400
            )

        half = 1.959964 * np.sqrt(0.05)
        step = grid[1] - grid[0]
        # At Mbar = 0 with flat pre-trends the set is the usual interval,
        # located to within one grid step.
        assert lo == pytest.approx(5.0 - half, abs=step)
        assert hi == pytest.approx(5.0 + half, abs=step)

    def test_default_grid_is_untouched_when_it_contains_the_estimate(self):
        from statspai.did._arp import rm_confidence_set

        sigma = np.diag([1e-4, 1e-4, 0.05])
        beta = np.array([0.0, 0.0, 1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, grid, _ = rm_confidence_set(
                beta, sigma, 2, 1, 0.0, method="C-LF", grid_points=200
            )
        assert grid[0] == pytest.approx(-20 * np.sqrt(0.05))
        assert grid[-1] == pytest.approx(20 * np.sqrt(0.05))
