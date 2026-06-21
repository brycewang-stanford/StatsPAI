"""Coverage tests for synth comparison, reporting, exports, and sensitivity."""

import importlib
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import statspai as sp

compare = importlib.import_module("statspai.synth.compare")
report = importlib.import_module("statspai.synth.report")
exports = importlib.import_module("statspai.synth.exports")
sensitivity = importlib.import_module("statspai.synth.sensitivity")


def _panel(n_units=9, n_periods=18, treatment_time=12, effect=4.0, seed=41):
    rng = np.random.default_rng(seed)
    alphas = rng.normal(10, 2, n_units)
    betas = rng.normal(0.5, 0.1, n_units)
    common = rng.normal(0, 0.4, n_periods)
    records = []
    for i in range(n_units):
        for ti, t in enumerate(range(1, n_periods + 1)):
            y = alphas[i] + betas[i] * t + common[ti] + rng.normal(0, 0.2)
            if i == 0 and t >= treatment_time:
                y += effect
            records.append({"unit": f"u{i}", "time": t, "outcome": y})
    return pd.DataFrame(records)


COMMON = dict(
    outcome="outcome", unit="unit", time="time", treated_unit="u0", treatment_time=12
)


@pytest.fixture
def panel():
    return _panel()


@pytest.fixture
def classic_result(panel):
    return sp.synth(panel, **COMMON, method="classic", placebo=True)


# ---------------- compare ----------------
def test_synth_compare_default(panel):
    comp = compare.synth_compare(
        panel, **COMMON, methods=["classic", "ridge"], placebo=False
    )
    assert hasattr(comp, "summary")
    s = comp.summary()
    assert isinstance(s, str) and len(s) > 0
    assert isinstance(repr(comp), str)
    assert isinstance(str(comp), str)


def test_synth_compare_table_and_recommend(panel):
    comp = compare.synth_compare(
        panel, **COMMON, methods=["classic", "demeaned", "ridge"], placebo=False
    )
    assert isinstance(comp.comparison_table, pd.DataFrame)
    assert len(comp.comparison_table) >= 1
    assert comp.recommended in comp.results
    # latex / markdown export from comparison object
    assert isinstance(comp.to_latex(), str)
    assert isinstance(comp.to_markdown(), str)


def test_synth_recommend(panel):
    rec = compare.synth_recommend(panel, **COMMON)
    assert isinstance(rec, str) and len(rec) > 0


def test_synth_compare_excel(panel):
    comp = compare.synth_compare(
        panel, **COMMON, methods=["classic", "ridge"], placebo=False
    )
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cmp.xlsx")
        out = comp.to_excel(path)
        assert os.path.exists(out)


# ---------------- report ----------------
@pytest.mark.parametrize("output", ["text", "markdown", "latex"])
def test_synth_report_formats(panel, output):
    rep = report.synth_report(
        panel, **COMMON, method="classic", output=output, sensitivity=False
    )
    assert isinstance(rep, str) and len(rep) > 0


def test_synth_report_with_sensitivity(panel):
    rep = report.synth_report(
        panel, **COMMON, method="classic", output="text", sensitivity=True
    )
    assert isinstance(rep, str) and len(rep) > 0


def test_synth_report_to_file(panel):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "report.md")
        report.synth_report_to_file(
            panel,
            **COMMON,
            method="classic",
            output="markdown",
            sensitivity=False,
            filename=path,
        )
        assert os.path.exists(path)


# ---------------- exports ----------------
def test_synth_to_latex_single(classic_result):
    tex = exports.synth_to_latex(
        classic_result,
        caption="Test",
        label="tab:test",
        show_weights=True,
        top_n_weights=3,
    )
    assert "\\begin{table}" in tex or "tabular" in tex


def test_synth_to_markdown_single(classic_result):
    md = exports.synth_to_markdown(
        classic_result, title="SC results", show_weights=True, top_n_weights=3
    )
    assert isinstance(md, str) and "|" in md


def test_synth_to_latex_list(panel):
    r1 = sp.synth(panel, **COMMON, method="classic", placebo=False)
    r2 = sp.synth(panel, **COMMON, method="ridge", placebo=False)
    tex = exports.synth_to_latex(
        [r1, r2], method_names=["Classic", "Ridge"], show_ci=True
    )
    assert "tabular" in tex
    md = exports.synth_to_markdown([r1, r2], method_names=["Classic", "Ridge"])
    assert "|" in md


def test_synth_to_excel(classic_result):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "out.xlsx")
        out = exports.synth_to_excel(classic_result, path)
        assert os.path.exists(out)


def test_synth_to_latex_no_ci(classic_result):
    tex = exports.synth_to_latex(classic_result, show_ci=False, booktabs=False)
    assert "tabular" in tex


# ---------------- sensitivity ----------------
def test_synth_loo(panel):
    df = sensitivity.synth_loo(panel, **COMMON)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1


def test_synth_time_placebo(panel):
    df = sensitivity.synth_time_placebo(panel, **COMMON, n_placebo_times=3)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1


def test_synth_donor_sensitivity(panel):
    df = sensitivity.synth_donor_sensitivity(panel, **COMMON, k=2, n_samples=20, seed=0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1


def test_synth_rmspe_filter(panel):
    df = sensitivity.synth_rmspe_filter(panel, **COMMON, thresholds=[2.0, 5.0, 20.0])
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1


def test_synth_sensitivity_suite(panel):
    out = sensitivity.synth_sensitivity(panel, **COMMON, n_donor_samples=20, seed=0)
    assert isinstance(out, dict)
    assert len(out) > 0
