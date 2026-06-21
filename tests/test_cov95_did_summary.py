"""Coverage tests for statspai.did.summary (did_summary + exporters + report)."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult


@pytest.fixture(scope="module")
def stag():
    return sp.dgp_did(n_units=120, n_periods=8, staggered=True, seed=0)


def _cols(df):
    # discover the column names used by dgp_did
    return df.columns.tolist()


def test_did_summary_single_method(stag):
    cols = _cols(stag)
    # dgp_did standard columns: unit, time, first_treat, y
    out = sp.did_summary(
        stag, y="y", time="time", first_treat="first_treat", group="unit", methods="cs"
    )
    assert isinstance(out.detail, pd.DataFrame)
    assert (
        "cs" in out.model_info["methods_fit"]
        or "cs" in out.model_info["methods_failed"]
    )
    txt = out.summary()
    assert "DID Method-Robustness Summary" in txt


def test_did_summary_multi_method(stag):
    out = sp.did_summary(
        stag,
        y="y",
        time="time",
        first_treat="first_treat",
        group="unit",
        methods=["cs", "sa", "bjs"],
    )
    assert len(out.detail) == 3
    # at least one fit
    assert len(out.model_info["methods_fit"]) >= 1
    txt = out.summary()
    assert "Fitted methods" in txt or "Failed" in txt


def test_did_summary_auto(stag):
    out = sp.did_summary(
        stag,
        y="y",
        time="time",
        first_treat="first_treat",
        group="unit",
        methods="auto",
    )
    assert len(out.detail) == 5


def test_did_summary_unknown_method(stag):
    with pytest.raises(ValueError):
        sp.did_summary(
            stag,
            y="y",
            time="time",
            first_treat="first_treat",
            group="unit",
            methods=["cs", "not_a_method"],
        )


def test_did_summary_missing_column(stag):
    with pytest.raises(KeyError):
        sp.did_summary(
            stag,
            y="nope",
            time="time",
            first_treat="first_treat",
            group="unit",
            methods="cs",
        )


def test_did_summary_verbose(stag, capsys):
    sp.did_summary(
        stag,
        y="y",
        time="time",
        first_treat="first_treat",
        group="unit",
        methods="cs",
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "running cs" in captured.out


def test_did_summary_with_sensitivity(stag):
    out = sp.did_summary(
        stag,
        y="y",
        time="time",
        first_treat="first_treat",
        group="unit",
        methods=["cs", "sa"],
        include_sensitivity=True,
    )
    # breakdown_m column present
    assert "breakdown_m" in out.detail.columns


def test_did_summary_to_markdown(stag):
    out = sp.did_summary(
        stag,
        y="y",
        time="time",
        first_treat="first_treat",
        group="unit",
        methods=["cs", "sa"],
    )
    md = sp.did_summary_to_markdown(out)
    assert "| Method |" in md
    assert "Mean across methods" in md
    # variant flags
    md2 = sp.did_summary_to_markdown(out, include_ci=False)
    assert "95% CI" not in md2


def test_did_summary_to_latex(stag):
    out = sp.did_summary(
        stag,
        y="y",
        time="time",
        first_treat="first_treat",
        group="unit",
        methods=["cs", "sa"],
    )
    tex = sp.did_summary_to_latex(out)
    assert "\\begin{table}" in tex
    assert "\\toprule" in tex
    tex2 = sp.did_summary_to_latex(out, include_ci=False, caption="X", label="tab:x")
    assert "tab:x" in tex2


def test_exporters_reject_non_summary_result():
    r = CausalResult(
        method="x",
        estimand="ATT",
        estimate=1.0,
        se=0.1,
        pvalue=0.5,
        ci=(0, 1),
        alpha=0.05,
        n_obs=10,
        detail=None,
        model_info={},
    )
    with pytest.raises(ValueError):
        sp.did_summary_to_markdown(r)
    with pytest.raises(ValueError):
        sp.did_summary_to_latex(r)


def test_exporters_reject_malformed_detail():
    r = CausalResult(
        method="x",
        estimand="ATT",
        estimate=1.0,
        se=0.1,
        pvalue=0.5,
        ci=(0, 1),
        alpha=0.05,
        n_obs=10,
        detail=pd.DataFrame({"foo": [1]}),
        model_info={"_did_summary_marker": True},
    )
    with pytest.raises(ValueError):
        sp.did_summary_to_markdown(r)


def test_did_summary_markdown_with_breakdown(stag):
    out = sp.did_summary(
        stag,
        y="y",
        time="time",
        first_treat="first_treat",
        group="unit",
        methods=["cs"],
        include_sensitivity=True,
    )
    md = sp.did_summary_to_markdown(out, include_breakdown=True)
    tex = sp.did_summary_to_latex(out, include_breakdown=True)
    assert isinstance(md, str) and isinstance(tex, str)


def test_did_report_bundle(stag, tmp_path):
    out = sp.did_report(
        stag,
        y="y",
        time="time",
        first_treat="first_treat",
        group="unit",
        save_to=str(tmp_path),
        methods=["cs", "sa"],
        include_sensitivity=False,
    )
    assert (tmp_path / "did_summary.txt").exists()
    assert (tmp_path / "did_summary.md").exists()
    assert (tmp_path / "did_summary.tex").exists()
    assert (tmp_path / "did_summary.json").exists()
    assert out is not None
