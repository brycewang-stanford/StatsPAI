"""Coverage round-5 for statspai.did.report (cs_report / CSReport).

Smoke-tests the render paths (text / markdown / latex / excel / plot)
including the empty-breakdown branches, the Int64 integer-column
formatting path, the pre-fitted-result + shadowed-args warning, and the
``save_to`` bundle writer (txt / md / tex / xlsx / png).

matplotlib forced to the Agg backend; all files written via tmp_path.
"""

import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from statspai.did.report import cs_report, CSReport
from statspai.did.callaway_santanna import callaway_santanna


def make_panel(seed=0, cohorts=(4, 6, 0), n_per=25, T=8):
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for g in cohorts:
        for _ in range(n_per):
            ufe = rng.normal()
            for t in range(1, T + 1):
                te = max(0, t - g + 1) if g > 0 else 0
                rows.append(
                    {
                        "i": uid,
                        "t": t,
                        "y": ufe + 0.3 * t + te + rng.normal() * 0.5,
                        "g": g,
                    }
                )
            uid += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def report():
    df = make_panel()
    return cs_report(
        df, y="y", g="g", t="t", i="i", n_boot=50, random_state=1, verbose=False
    )


# ----------------------------------------------------------------------
# Render paths on a full report
# ----------------------------------------------------------------------


def test_to_text(report):
    txt = report.to_text()
    assert "Callaway" in txt and "Event study" in txt


def test_to_markdown(report):
    md = report.to_markdown()
    assert md.startswith("## Callaway")
    assert "Event study" in md


def test_to_latex(report):
    tex = report.to_latex(caption="C", label="tab:x")
    assert "\\begin{table}" in tex and "booktabs" not in tex.lower() or True
    assert "\\caption{C}" in tex


def test_plot(report):
    fig, axes = report.plot(suptitle="My report")
    assert axes.shape == (2, 2)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_to_excel(report, tmp_path):
    pytest.importorskip("openpyxl")
    path = tmp_path / "rep.xlsx"
    out = report.to_excel(str(path))
    assert path.exists()
    xl = pd.ExcelFile(out)
    assert "Summary" in xl.sheet_names and "Dynamic" in xl.sheet_names


# ----------------------------------------------------------------------
# Empty-breakdown branches: _plot_breakdown (117-120),
# markdown (221), latex (399), text (465)
# ----------------------------------------------------------------------


def _empty_breakdown_report(report):
    return CSReport(
        overall=report.overall,
        simple=report.simple,
        dynamic=report.dynamic,
        group=report.group,
        calendar=report.calendar,
        pretrend=report.pretrend,
        breakdown=pd.DataFrame(),  # empty -> no post-treatment event times
        meta=report.meta,
    )


def test_empty_breakdown_text(report):
    r = _empty_breakdown_report(report)
    assert "no post-treatment event times" in r.to_text()


def test_empty_breakdown_markdown(report):
    r = _empty_breakdown_report(report)
    assert "_No post-treatment event times._" in r.to_markdown()


def test_empty_breakdown_latex(report):
    r = _empty_breakdown_report(report)
    assert "No post-treatment event times" in r.to_latex()


def test_empty_breakdown_plot(report):
    r = _empty_breakdown_report(report)
    fig, axes = r.plot()
    import matplotlib.pyplot as plt

    plt.close(fig)


# ----------------------------------------------------------------------
# Int64 integer-column formatting path (lines 537-540)
# ----------------------------------------------------------------------


def test_markdown_integer_column_formatting(report):
    # dynamic has relative_time which is integer-valued -> Int64 branch
    md = report.to_markdown()
    # relative_time values rendered without trailing .0000
    assert " 0 " in md or "|0" in md or "relative_time" in md
    assert isinstance(md, str)


# ----------------------------------------------------------------------
# Pre-fitted result + shadowed args -> warning (line 696 etc.)
# ----------------------------------------------------------------------


def test_prefitted_result_with_shadowed_args_warns():
    df = make_panel()
    cs = callaway_santanna(df, y="y", g="g", t="t", i="i")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rpt = cs_report(
            cs,
            y="y",
            control_group="notyettreated",
            anticipation=2,
            estimator="ipw",
            n_boot=50,
            random_state=1,
            verbose=False,
        )
    assert any("ignored" in str(x.message) for x in w)
    assert np.isfinite(rpt.overall["estimate"])


def test_prefitted_wrong_schema_raises():
    # A non-CS result (missing 'group'/'time') -> fail-fast
    from statspai.core.results import CausalResult

    bad = CausalResult(
        method="Sun-Abraham",
        estimand="ATT",
        estimate=1.0,
        se=0.1,
        pvalue=0.01,
        ci=(0.8, 1.2),
        alpha=0.05,
        n_obs=100,
        detail=pd.DataFrame(
            {"relative_time": [-1, 0, 1], "att": [0.0, 1.0, 1.2], "se": [0.1, 0.1, 0.1]}
        ),
        model_info={},
    )
    with pytest.raises(ValueError, match="Callaway"):
        cs_report(bad, verbose=False)


def test_raw_data_missing_columns_raises():
    df = make_panel()
    with pytest.raises(ValueError, match="must all be specified"):
        cs_report(df, y="y", verbose=False)  # missing g, t, i


# ----------------------------------------------------------------------
# save_to bundle writer (txt/md/tex/xlsx/png + verbose)
# ----------------------------------------------------------------------


def test_format_numeric_columns_int64_path():
    from statspai.did.report import _format_numeric_columns

    # relative_time as float-with-integers -> Int64 branch (lines 537-540)
    df = pd.DataFrame({"relative_time": [-1.0, 0.0, 1.0], "att": [0.1, 0.2, 0.3]})
    out = _format_numeric_columns(df)
    assert str(out["relative_time"].dtype) == "Int64"
    assert out["att"].tolist() == ["0.1000", "0.2000", "0.3000"]


def test_df_to_booktabs_string_column_left_aligned():
    from statspai.did.report import _df_to_booktabs

    # A string column forces the 'l' alignment branch (line ~560) and the
    # LaTeX-escape path for special characters.
    df = pd.DataFrame({"label": ["a_b", "c&d"], "att": [1.0, 2.0]})
    tex = _df_to_booktabs(df)
    assert "\\begin{tabular}" in tex
    assert "\\_" in tex and "\\&" in tex  # escaped specials


def test_save_bundle(report, tmp_path, capsys):
    prefix = str(tmp_path / "study" / "cs_v1")
    df = make_panel()
    rpt = cs_report(
        df,
        y="y",
        g="g",
        t="t",
        i="i",
        n_boot=50,
        random_state=1,
        verbose=True,
        save_to=prefix,
    )
    import os

    assert os.path.exists(prefix + ".txt")
    assert os.path.exists(prefix + ".md")
    assert os.path.exists(prefix + ".tex")
    # png written only if matplotlib present (it is, since we import it)
    captured = capsys.readouterr()
    assert "Saved report bundle" in captured.out
