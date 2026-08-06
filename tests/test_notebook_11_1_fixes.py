"""Regression cover for defects surfaced by the RCT / quasi-experiment
teaching notebook (11.1).

Each class pins one defect that produced wrong or unusable output while
the notebook itself ran without error:

* Durbin-Wu-Hausman returned NaN (binary treatment) or a silently wrong
  statistic (integer counts) because a float residual was assigned into
  an integer buffer.
* ``tidy()`` labelled every event-study row ``att(g=,t=)``, discarding
  the event time.
* ``summary()`` printed whole pandas Series row-by-row — one SDID
  summary ran to 110 lines.
* ``love_plot`` could not consume the matching result that produced it.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


# ----------------------------------------------------------------------
# Durbin-Wu-Hausman dtype truncation
# ----------------------------------------------------------------------


class TestHausmanDtype:
    def _lottery(self, dtype):
        rng = np.random.default_rng(2026)
        n = 5000
        grit = rng.normal(0, 1, n)
        z = rng.integers(0, 2, n)
        d = ((0.9 * z + 0.8 * grit + rng.normal(0, 1, n)) > 0.8).astype(dtype)
        y = 3 + 3.5 * d + 2.0 * grit + rng.normal(0, 1, n)
        return pd.DataFrame({"y": y, "d": d, "z": z})

    def test_binary_int_treatment_is_not_nan(self):
        """The headline symptom: a 0/1 int treatment gave NaN."""
        res = sp.ivreg("y ~ (d ~ z)", self._lottery(int))
        f = res.diagnostics["Hausman F-stat"]
        p = res.diagnostics["Hausman p-value"]
        assert np.isfinite(f), "DWH F-statistic must not be NaN"
        assert np.isfinite(p)
        assert f > 0

    def test_int_and_float_treatment_agree(self):
        int_f = sp.ivreg("y ~ (d ~ z)", self._lottery(int)).diagnostics[
            "Hausman F-stat"
        ]
        flt_f = sp.ivreg("y ~ (d ~ z)", self._lottery(float)).diagnostics[
            "Hausman F-stat"
        ]
        assert int_f == pytest.approx(flt_f, rel=1e-9)

    def test_integer_counts_are_not_silently_wrong(self):
        """The dangerous variant: integer-valued counts produced a
        finite but wrong statistic (no NaN to warn the user)."""
        from statspai.regression.iv import _hausman_test

        rng = np.random.default_rng(0)
        n = 3000
        z = rng.normal(0, 1, n)
        u = rng.normal(0, 1, n)
        x_int = np.round(5 * z + 3 * u + rng.normal(0, 1, n)).astype(int)
        y = 2 + 1.5 * x_int + 2 * u + rng.normal(0, 1, n)
        X_exog = np.ones((n, 1))
        W = np.column_stack([np.ones(n), z])
        as_int = _hausman_test(y, X_exog, x_int[:, None], W)["statistic"]
        as_float = _hausman_test(y, X_exog, x_int[:, None].astype(float), W)[
            "statistic"
        ]
        assert as_int == pytest.approx(as_float, rel=1e-12)

    def test_float_path_unchanged(self):
        """Guard against a regression in the already-correct path."""
        from statspai.regression.iv import _hausman_test

        rng = np.random.default_rng(1)
        n = 2000
        z = rng.normal(size=n)
        u = rng.normal(size=n)
        x = z + u + rng.normal(size=n)
        y = x + u + rng.normal(size=n)
        stat = _hausman_test(
            y, np.ones((n, 1)), x[:, None], np.column_stack([np.ones(n), z])
        )["statistic"]
        assert stat == pytest.approx(252.63791, rel=1e-4)


# ----------------------------------------------------------------------
# tidy() labels for aggregated DiD results
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def cs_result():
    mp = sp.datasets.mpdta()
    return sp.callaway_santanna(
        data=mp, y="lemp", t="year", i="countyreal", g="first_treat"
    )


class TestAggregatedTidyLabels:
    def test_dynamic_carries_event_time(self, cs_result):
        dyn = sp.aggte(cs_result, type="dynamic", bstrap=False)
        tidy = dyn.tidy()
        terms = [t for t in tidy["term"] if t != "ATT"]
        assert terms, "dynamic aggregation should emit per-period rows"
        assert not any(
            t == "att(g=,t=)" for t in terms
        ), "event-study rows must not collapse to an empty group-time label"
        assert all(t.startswith("event_") for t in terms), terms
        # Event times must round-trip from the detail frame.
        expected = [f"event_{int(e):+d}" for e in dyn.detail["relative_time"]]
        assert terms == expected
        assert set(tidy["type"]) == {"main", "event_study"}

    def test_group_aggregation_labels_group(self, cs_result):
        grp = sp.aggte(cs_result, type="group", bstrap=False)
        terms = [t for t in grp.tidy()["term"] if t != "ATT"]
        assert all(t.startswith("att(g=") for t in terms), terms

    def test_calendar_aggregation_labels_time(self, cs_result):
        cal = sp.aggte(cs_result, type="calendar", bstrap=False)
        terms = [t for t in cal.tidy()["term"] if t != "ATT"]
        assert all(t.startswith("att(t=") for t in terms), terms

    def test_group_time_result_unchanged(self, cs_result):
        """The full group-time table keeps its att(g=,t=) labels."""
        tidy = cs_result.tidy()
        gt = tidy[tidy["type"] == "group_time"]["term"].tolist()
        assert gt, "group-time rows should still be present"
        assert all(t.startswith("att(g=") and ",t=" in t for t in gt), gt[:3]


# ----------------------------------------------------------------------
# summary() footer: vector dumps and solver telemetry
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def prop99():
    return sp.datasets.california_prop99()


class TestSummaryFooter:
    def test_sdid_summary_is_compact(self, prop99):
        sd = sp.sdid(
            data=prop99,
            outcome="cigsale",
            unit="state",
            time="year",
            treated_unit="California",
            treatment_time=1989,
        )
        text = sd.summary()
        n_lines = text.count("\n") + 1
        assert n_lines < 40, f"SDID summary ran to {n_lines} lines"
        # Vector entries are summarised, not dumped row-by-row.
        assert "values; see .model_info" in text
        assert "dtype: float64" not in text
        # The data is still reachable programmatically.
        assert len(sd.model_info["time_weights"]) == 19

    def test_scm_hides_solver_telemetry_but_discloses_count(self, prop99):
        scm = sp.synth(
            data=prop99,
            outcome="cigsale",
            unit="state",
            time="year",
            treated_unit="California",
            treatment_time=1989,
        )
        text = scm.summary()
        assert "Solver Near Best Weight L1 Max" not in text
        assert "solver/diagnostic entries hidden" in text
        # The substantive non-uniqueness disclosure stays visible.
        assert "Weight Solution Nonunique" in text
        # And the hidden entries remain in model_info.
        assert "solver_best_loss" in scm.model_info

    def test_headline_numbers_unchanged(self, prop99):
        scm = sp.synth(
            data=prop99,
            outcome="cigsale",
            unit="state",
            time="year",
            treated_unit="California",
            treatment_time=1989,
        )
        assert scm.estimate == pytest.approx(-19.760529, rel=1e-5)


# ----------------------------------------------------------------------
# love_plot composability
# ----------------------------------------------------------------------


class TestLovePlotComposability:
    COVS = ["age", "educ", "black", "hispanic", "married", "nodegree"]

    @pytest.fixture(scope="class")
    def matched(self):
        nsw = sp.datasets.nsw_lalonde()
        return nsw, sp.psmatch2(
            nsw, treat="treat", outcome="re78", covariates=self.COVS
        )

    def test_accepts_matching_result(self, matched):
        _, m = matched
        fig, ax = sp.love_plot(m)
        assert len(ax.get_yticklabels()) == len(self.COVS)

    def test_raw_frame_path_unchanged(self, matched):
        nsw, _ = matched
        fig, ax = sp.love_plot(nsw, treatment="treat", covariates=self.COVS)
        assert len(ax.get_yticklabels()) == len(self.COVS)

    def test_explicit_args_override_result(self, matched):
        _, m = matched
        subset = self.COVS[:3]
        fig, ax = sp.love_plot(m, covariates=subset)
        assert len(ax.get_yticklabels()) == len(subset)

    def test_frame_without_spec_raises(self, matched):
        nsw, _ = matched
        from statspai.exceptions import MethodIncompatibility

        with pytest.raises(MethodIncompatibility, match="requires treatment"):
            sp.love_plot(nsw)

    def test_unusable_object_raises(self):
        from statspai.exceptions import MethodIncompatibility

        with pytest.raises(MethodIncompatibility, match="cannot read"):
            sp.love_plot(42)


# ----------------------------------------------------------------------
# psmatch2 summary wording
# ----------------------------------------------------------------------


def test_psmatch2_common_support_wording():
    """'none' means no restriction was imposed, not 'no overlap found'."""
    nsw = sp.datasets.nsw_lalonde()
    m = sp.psmatch2(
        nsw, treat="treat", outcome="re78", covariates=["age", "educ", "re74"]
    )
    text = m.summary()
    assert "Common support    : no restriction imposed" in text
