"""Extra coverage for statspai.did.summary: failed-method NaN rows in markdown
& latex, _stars branches, breakdown-print in summary(), did_report verbose+PNG."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did import summary as summod


def _stars_inputs():
    return summod._stars


def test_stars_branches():
    f = summod._stars
    assert f(0.005) == "***"
    assert f(0.03) == "**"
    assert f(0.07) == "*"
    assert f(0.5) == ""
    assert f(np.nan) == ""


@pytest.fixture(scope="module")
def stag():
    return sp.dgp_did(n_units=100, n_periods=8, staggered=True, seed=1)


def _force_failure_df(stag):
    # Make one method fail by giving a tiny / degenerate panel for one method
    # is hard; instead inject a NaN row by monkey-running a method on data that
    # makes a sub-estimator raise. Simpler: run multi-method on data where the
    # 'stacked' estimator can fail on a too-short panel. We rely on the public
    # NaN-handling path being reachable when an estimator raises an expected
    # exception. If all fit, we still cover the success path.
    return stag


def test_summary_markdown_with_failed_method(monkeypatch, stag):
    # Force the 'sa' runner to raise so its row is NaN -> markdown NA branch.
    def _boom(*a, **k):
        raise ValueError("synthetic failure for coverage")

    monkeypatch.setitem(summod._DISPATCH, "sa", _boom)
    out = sp.did_summary(
        stag,
        y="y",
        time="time",
        first_treat="first_treat",
        group="unit",
        methods=["cs", "sa"],
    )
    assert "sa" in out.model_info["methods_failed"]
    md = sp.did_summary_to_markdown(out)
    assert "—" in md  # NA dash row rendered
    tex = sp.did_summary_to_latex(out)
    assert "---" in tex  # NA dashes in latex
    txt = out.summary()
    assert "Failed" in txt


def test_summary_breakdown_print(stag):
    out = sp.did_summary(
        stag,
        y="y",
        time="time",
        first_treat="first_treat",
        group="unit",
        methods=["cs"],
        include_sensitivity=True,
    )
    txt = out.summary()
    # if breakdown computed, the summary prints "Breakdown M*"
    if out.model_info.get("breakdown_m") is not None:
        assert "Breakdown M*" in txt


def test_did_report_verbose_with_png(stag, tmp_path):
    out = sp.did_report(
        stag,
        y="y",
        time="time",
        first_treat="first_treat",
        group="unit",
        save_to=str(tmp_path),
        methods=["cs", "sa"],
        include_sensitivity=False,
        verbose=True,
    )
    # PNG written when matplotlib present
    import importlib.util

    if importlib.util.find_spec("matplotlib") is not None:
        assert (tmp_path / "did_summary.png").exists()
    assert (tmp_path / "did_summary.json").exists()
    assert out is not None
