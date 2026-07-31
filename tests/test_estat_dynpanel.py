"""``sp.estat`` postestimation for dynamic-panel GMM fits.

Stata splits this across ``estat abond``, ``estat sargan`` and (in
``xtabond2``) an always-printed difference-in-Hansen block. StatsPAI
computes all three during the fit, so these handlers *present* stored
numbers rather than recomputing them — a postestimation command that
re-derives a statistic can silently disagree with the fit it reports on.
The tests below therefore check the presentation contract and that the
numbers are the fit's own, not that the statistics are correct: their
correctness is pinned against Stata in
``tests/reference_parity/test_dynpanel_abdata_parity.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

ABDATA = (
    Path(__file__).parent / "reference_parity" / "_fixtures" / "dynpanel_abdata.csv"
)

pytestmark = pytest.mark.skipif(
    not ABDATA.exists(), reason="abdata fixture not generated"
)


@pytest.fixture(scope="module")
def system_fit():
    df = pd.read_csv(ABDATA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.xtabond(
            df, y="n", x=["w", "k"], id="id", time="year", lags=1, method="system"
        )


@pytest.fixture(scope="module")
def diff_fit():
    df = pd.read_csv(ABDATA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.xtabond(df, y="n", x=["w", "k"], id="id", time="year", lags=1)


class TestAbond:
    def test_reports_both_orders(self, system_fit):
        out = sp.estat(system_fit, "abond", print_results=False)
        assert [row["order"] for row in out["rows"]] == [1, 2]

    def test_numbers_are_the_fit_s_own(self, system_fit):
        """No recomputation: the values must be identical, not merely close."""
        out = sp.estat(system_fit, "abond", print_results=False)
        info = system_fit.model_info
        assert out["rows"][0]["z"] == info["ar1_z"]
        assert out["rows"][1]["z"] == info["ar2_z"]
        assert out["rows"][0]["pvalue"] == info["ar1_p"]

    def test_interpretation_keys_off_ar2(self, system_fit):
        """AR(1) rejecting is expected; only AR(2) drives the verdict."""
        out = sp.estat(system_fit, "abond", print_results=False)
        assert out["rows"][0]["reject"] is True, "AR(1) should reject on abdata"
        assert out["rows"][1]["reject"] is False
        assert "AR(2) does not reject" in out["interpretation"]

    def test_aliases_resolve(self, system_fit):
        direct = sp.estat(system_fit, "abond", print_results=False)
        for alias in ("arellano_bond", "ar", "ABOND"):
            assert sp.estat(system_fit, alias, print_results=False) == direct


class TestSargan:
    def test_reports_sargan_and_hansen_separately(self, system_fit):
        """They answer the same question under different assumptions."""
        out = sp.estat(system_fit, "sargan", print_results=False)
        names = [row["name"] for row in out["rows"]]
        assert names == ["Sargan", "Hansen J"]
        robust = {
            row["name"]: row["robust_to_heteroskedasticity"] for row in out["rows"]
        }
        assert robust == {"Sargan": False, "Hansen J": True}

    def test_values_match_the_fit(self, system_fit):
        out = sp.estat(system_fit, "sargan", print_results=False)
        info = system_fit.model_info
        assert out["rows"][0]["statistic"] == info["sargan_stat"]
        assert out["rows"][1]["statistic"] == info["hansen_stat"]
        assert out["n_instruments"] == info["n_instruments"]

    def test_warns_in_prose_when_instruments_swamp_units(self):
        """A Hansen p near 1 with many instruments is not reassurance."""
        rng = np.random.default_rng(4)
        rows = []
        for i in range(15):
            a = rng.normal()
            y = a / 0.5 + rng.normal()
            for _ in range(10):
                y = 0.5 * y + a + rng.normal()
            for t in range(14):
                y = 0.5 * y + a + rng.normal()
                rows.append({"id": i, "time": t, "y": y})
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sp.xtabond(df, y="y", id="id", time="time", lags=1)
        out = sp.estat(fit, "sargan", print_results=False)
        assert out["n_instruments"] >= out["n_units"]
        assert "unreliable at this ratio" in out["interpretation"]


class TestDifferenceInHansen:
    def test_lists_every_instrument_subset(self, system_fit):
        out = sp.estat(system_fit, "difhansen", print_results=False)
        subsets = {row["subset"] for row in out["rows"]}
        assert "GMM instruments for levels" in subsets
        assert any(s.startswith("iv(") for s in subsets)

    def test_values_match_the_fit(self, system_fit):
        out = sp.estat(system_fit, "difhansen", print_results=False)
        stored = system_fit.model_info["difference_in_hansen"]
        for row in out["rows"]:
            assert row["statistic"] == stored[row["subset"]]["statistic"]

    def test_difference_gmm_has_no_level_subset(self, diff_fit):
        out = sp.estat(diff_fit, "difhansen", print_results=False)
        assert not any(
            row["subset"] == "GMM instruments for levels" for row in out["rows"]
        )


class TestDispatch:
    def test_all_returns_the_three_dynamic_panel_tests(self, system_fit):
        """'all' means the regression diagnostics for OLS and these for GMM.

        A dynamic-panel fit has no fitted values in levels, so the ordinary
        ``estat all`` battery is undefined for it — before this it returned
        "no applicable tests", which is worse than useless on a fit that has
        three well-defined ones.
        """
        out = sp.estat(system_fit, "all", print_results=False)
        assert isinstance(out, list)
        assert [item["test"] for item in out] == ["abond", "sargan", "difhansen"]

    def test_rejects_non_dynamic_panel_results(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(size=120)})
        df["y"] = 1 + 0.5 * df["x"] + rng.normal(size=120)
        ols = sp.regress("y ~ x", data=df)
        with pytest.raises(ValueError, match="dynamic-panel GMM fits"):
            sp.estat(ols, "abond", print_results=False)

    def test_ordinary_results_keep_the_ordinary_all(self):
        """The dynamic-panel branch must not capture OLS fits."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"x": rng.normal(size=150)})
        df["y"] = 1 + 0.5 * df["x"] + rng.normal(size=150)
        ols = sp.regress("y ~ x", data=df)
        out = sp.estat(ols, "all", print_results=False)
        assert isinstance(out, list)
        assert not any(item.get("test") == "abond" for item in out)

    def test_printing_does_not_raise(self, system_fit, capsys):
        sp.estat(system_fit, "all")
        printed = capsys.readouterr().out
        assert "Arellano-Bond test for serial correlation" in printed
        assert "Over-identification tests" in printed
        assert "Difference-in-Hansen" in printed
        # The tables must actually carry numbers, not just banners.
        assert "Hansen J" in printed and "instruments:" in printed
