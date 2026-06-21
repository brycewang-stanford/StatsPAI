"""Round-4 coverage margin: synth.report formatter branches.

Drives ``sp.synth_report`` over real synthetic-control panels in all
three output formats (text / markdown / latex), with the sensitivity
section both on and off, and with a large donor pool / many post
periods so the truncation and quality branches fire.

All output is via ``tmp_path`` -- never the repo root.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.synth import report as _report


@pytest.fixture(scope="module")
def ca():
    return sp.california_tobacco()


@pytest.fixture(scope="module")
def germany():
    return sp.german_reunification()


@pytest.mark.parametrize("output", ["text", "markdown", "latex"])
def test_synth_report_all_formats_with_sensitivity(ca, output):
    rep = sp.synth_report(
        data=ca,
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
        output=output,
        sensitivity=True,
        n_donor_samples=10,
        sensitivity_seed=0,
    )
    assert isinstance(rep, str) and len(rep) > 0
    # Sensitivity section present -> citation lands in section 8.
    assert "8." in rep or "Citation" in rep or "CITATION" in rep


@pytest.mark.parametrize("output", ["text", "markdown", "latex"])
def test_synth_report_all_formats_no_sensitivity(ca, output):
    rep = sp.synth_report(
        data=ca,
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
        output=output,
        sensitivity=False,
    )
    assert isinstance(rep, str)
    # No sensitivity -> citation section is 7.
    assert "7." in rep or "Citation" in rep or "CITATION" in rep


@pytest.mark.parametrize("output", ["text", "markdown", "latex"])
def test_synth_report_truncation_branches(ca, output):
    # Early treatment time => many post periods (>20) and full donor pool
    # (>15) so the "... N more" truncation branches and the per-period /
    # weight tables all execute.
    rep = sp.synth_report(
        data=ca,
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1976,
        output=output,
        sensitivity=False,
    )
    assert isinstance(rep, str) and len(rep) > 0


@pytest.mark.parametrize("output", ["text", "markdown", "latex"])
def test_synth_report_germany_excellent_fit(germany, output):
    # West Germany fits extremely well (~1.5% of outcome SD) -> the
    # "Excellent" quality bucket in every formatter.
    rep = sp.synth_report(
        data=germany,
        outcome="gdppc",
        unit="country",
        time="year",
        treated_unit="West Germany",
        treatment_time=1990,
        output=output,
        sensitivity=False,
    )
    assert isinstance(rep, str)
    assert "Excellent" in rep


def test_synth_report_to_file_writes(tmp_path, ca):
    out = tmp_path / "ca_report.md"
    rep = sp.synth_report_to_file(
        data=ca,
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
        output="markdown",
        sensitivity=False,
        filename=str(out),
    )
    assert out.exists()
    assert out.read_text(encoding="utf-8") == rep


def test_synth_report_invalid_output_raises(ca):
    with pytest.raises(ValueError, match="output must be"):
        sp.synth_report(
            data=ca,
            outcome="cigsale",
            unit="state",
            time="year",
            treated_unit="California",
            treatment_time=1989,
            output="json",
        )


def test_canonicalise_mi_treated_units_list_and_treat_time():
    """Exercise the SDID-style key backfill in _canonicalise_mi."""

    class _Res:
        def __init__(self):
            self.model_info = {
                "treated_units": ["A"],
                "treat_time": 5,
                "T_pre": 4,
                "T_post": 3,
                "n_control": 6,
            }

    mi = _report._canonicalise_mi(_Res(), None, None)
    assert mi["treated_unit"] == "A"
    assert mi["treatment_time"] == 5
    assert mi["n_pre_periods"] == 4
    assert mi["n_post_periods"] == 3
    assert mi["n_donors"] == 6


def test_canonicalise_mi_treated_units_multi():
    class _Res:
        model_info = {"treated_units": ["A", "B"]}

    mi = _report._canonicalise_mi(_Res(), None, None)
    assert mi["treated_unit"] == "A, B"
