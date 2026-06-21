"""Tests for the publication-grade SCM exports + new plot options.

Covers:
* ``sp.synth_to_latex`` (single result + comparison)
* ``sp.synth_to_markdown``
* ``sp.synth_to_excel`` (multi-sheet workbook)
* ``SynthComparison.to_latex`` / ``to_markdown`` / ``to_excel`` methods
* ``synthplot(type='trajectory', pre_band=True, pi_band=True)``
* ``synthplot(type='gap', pre_band=True)``
* ``synth_report`` with ``method='sdid'`` (canonicalised model_info)
"""

from __future__ import annotations

import os
import tempfile

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import statspai as sp

# ====================================================================== #
#  Fixtures
# ====================================================================== #


@pytest.fixture(scope="module")
def calif():
    """California Prop99 panel — the canonical SCM benchmark."""
    return sp.california_tobacco()


@pytest.fixture(scope="module")
def classic_result(calif):
    return sp.synth(
        calif,
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
        method="classic",
        placebo=False,
    )


@pytest.fixture(scope="module")
def sdid_result(calif):
    return sp.sdid(
        calif,
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
    )


@pytest.fixture(scope="module")
def scpi_result(calif):
    return sp.scpi(
        calif,
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
    )


@pytest.fixture(scope="module")
def comparison(calif):
    return sp.synth_compare(
        calif,
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
        methods=["classic", "augmented", "sdid"],
        placebo=False,
    )


# ====================================================================== #
#  synth_to_latex
# ====================================================================== #


class TestSynthToLatex:

    def test_single_result_minimal(self, classic_result):
        latex = sp.synth_to_latex(classic_result)
        assert "\\begin{table}" in latex
        assert "\\end{table}" in latex
        assert "\\toprule" in latex
        assert "ATT" in latex
        assert "Pre-RMSPE" in latex

    def test_single_result_with_weights(self, classic_result):
        latex = sp.synth_to_latex(
            classic_result,
            show_weights=True,
            top_n_weights=3,
        )
        assert "Top donor weights" in latex
        # Classic SCM California study should pick Montana / Nevada
        assert "Montana" in latex or "Nevada" in latex

    def test_comparison(self, comparison):
        latex = sp.synth_to_latex(
            comparison,
            caption="Test",
            label="tab:test",
        )
        assert "\\caption{Test}" in latex
        assert "\\label{tab:test}" in latex
        assert "classic" in latex
        assert "augmented" in latex

    def test_comparison_method_via_object_method(self, comparison):
        latex = comparison.to_latex(show_weights=True)
        assert "Top donor weights" in latex
        assert "\\toprule" in latex

    def test_no_booktabs_falls_back_to_hline(self, classic_result):
        latex = sp.synth_to_latex(classic_result, booktabs=False)
        assert "\\toprule" not in latex
        assert "\\hline" in latex

    def test_invalid_input_raises(self):
        with pytest.raises(TypeError):
            sp.synth_to_latex("not a result")  # type: ignore[arg-type]


# ====================================================================== #
#  synth_to_markdown
# ====================================================================== #


class TestSynthToMarkdown:

    def test_single_result(self, classic_result):
        md = sp.synth_to_markdown(classic_result)
        assert "###" in md  # heading
        assert "**ATT**" in md
        assert "| Statistic |" in md  # pipe-table header

    def test_comparison_method(self, comparison):
        md = comparison.to_markdown(show_weights=True)
        assert "| classic |" in md
        assert "**Top donor weights:**" in md

    def test_significance_stars(self, comparison):
        md = comparison.to_markdown()
        # SDID should be highly significant on California Prop99
        assert "***" in md or "**" in md


# ====================================================================== #
#  synth_to_excel
# ====================================================================== #


class TestSynthToExcel:

    def test_single_result_writes_file(self, classic_result):
        openpyxl = pytest.importorskip("openpyxl")
        with tempfile.NamedTemporaryFile(
            suffix=".xlsx",
            delete=False,
        ) as f:
            path = sp.synth_to_excel(classic_result, f.name)
        try:
            assert os.path.exists(path)
            wb = openpyxl.load_workbook(path)
            assert "Summary" in wb.sheetnames
            assert "Weights" in wb.sheetnames
            assert "Diagnostics" in wb.sheetnames
            # Should have at least one Gap sheet for the single method
            assert any(s.startswith("Gap_") for s in wb.sheetnames)
        finally:
            os.unlink(path)

    def test_comparison_writes_multi_sheet(self, comparison):
        openpyxl = pytest.importorskip("openpyxl")
        with tempfile.NamedTemporaryFile(
            suffix=".xlsx",
            delete=False,
        ) as f:
            path = comparison.to_excel(f.name)
        try:
            wb = openpyxl.load_workbook(path)
            # One Gap sheet per method
            gap_sheets = [s for s in wb.sheetnames if s.startswith("Gap_")]
            assert len(gap_sheets) >= 2
            # Diagnostics has one row per method + header
            diag = wb["Diagnostics"]
            assert diag.max_row >= 4  # 3 methods + header
        finally:
            os.unlink(path)


# ====================================================================== #
#  Plot enhancements
# ====================================================================== #


class TestPlotOptions:

    def test_trajectory_with_pre_band(self, classic_result):
        fig, ax = sp.synthplot(classic_result, type="trajectory", pre_band=True)
        # Pre-band adds an additional fill_between collection
        assert any("pre-RMSPE" in (c.get_label() or "") for c in ax.collections)
        plt.close(fig)

    def test_trajectory_with_pi_band_scpi(self, scpi_result):
        fig, ax = sp.synthplot(scpi_result, type="trajectory", pi_band=True)
        labels = [c.get_label() or "" for c in ax.collections]
        assert any("PI on counterfactual" in lbl for lbl in labels)
        plt.close(fig)

    def test_gap_with_pre_band(self, classic_result):
        fig, ax = sp.synthplot(classic_result, type="gap", pre_band=True)
        labels = [c.get_label() or "" for c in ax.collections]
        assert any("pre-RMSPE" in lbl for lbl in labels)
        plt.close(fig)


# ====================================================================== #
#  synth_report SDID canonicalisation
# ====================================================================== #


class TestSynthReportSDID:

    def test_sdid_setup_table_no_NA(self, calif):
        md = sp.synth_report(
            calif,
            outcome="cigsale",
            unit="state",
            time="year",
            treated_unit="California",
            treatment_time=1989,
            method="sdid",
            output="markdown",
            sensitivity=False,
        )
        # Setup section should NOT be filled with N/A for SDID
        setup_block = md.split("## 2.")[0]
        assert "California" in setup_block
        assert "1989" in setup_block
        assert "Pre-treatment periods" in setup_block
        assert "| **Pre-treatment periods** | 19 |" in setup_block

    def test_sdid_pre_rmspe_recomputed(self, calif):
        md = sp.synth_report(
            calif,
            outcome="cigsale",
            unit="state",
            time="year",
            treated_unit="California",
            treatment_time=1989,
            method="sdid",
            output="markdown",
            sensitivity=False,
        )
        # Pre-RMSPE should be a finite number, not "nan"
        assert "Pre-RMSPE" in md
        assert "nan" not in md.split("Pre-RMSPE")[1].split("\n")[0].lower()

    def test_sdid_uses_arkhangelsky_citation(self, calif):
        md = sp.synth_report(
            calif,
            outcome="cigsale",
            unit="state",
            time="year",
            treated_unit="California",
            treatment_time=1989,
            method="sdid",
            output="markdown",
            sensitivity=False,
        )
        assert "Arkhangelsky" in md
        assert "[@arkhangelsky2021synthetic]" in md
