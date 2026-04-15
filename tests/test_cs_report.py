"""Tests for cs_report() — the one-call staggered-DID report."""
import numpy as np
import pandas as pd
import pytest

from statspai.did import callaway_santanna, cs_report, CSReport


def _staggered_panel(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(120):
        g = [3, 5, 7, 0][u // 30]
        ui = rng.normal(scale=0.3)
        for t in range(1, 9):
            te = max(0, t - g + 1) * 0.5 if g > 0 else 0
            rows.append({'i': u, 't': t, 'g': g,
                         'y': ui + 0.2 * t + te + rng.normal()})
    return pd.DataFrame(rows)


@pytest.fixture(scope='module')
def report():
    df = _staggered_panel(seed=42)
    return cs_report(df, y='y', g='g', t='t', i='i',
                     n_boot=300, random_state=0, verbose=False)


def test_report_is_dataclass_instance(report):
    assert isinstance(report, CSReport)


def test_report_exposes_all_aggregations(report):
    for attr, shape_min in [('simple', 1), ('dynamic', 1),
                            ('group', 1), ('calendar', 1)]:
        df = getattr(report, attr)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= shape_min
        assert {'att', 'se', 'ci_lower', 'ci_upper', 'pvalue'} <= set(df.columns)
    # Non-simple aggregations carry uniform-band columns.
    for attr in ('dynamic', 'group', 'calendar'):
        df = getattr(report, attr)
        assert {'cband_lower', 'cband_upper'} <= set(df.columns)


def test_report_breakdown_covers_all_post_times(report):
    dyn_post = report.dynamic[report.dynamic['relative_time'] >= 0]
    assert len(report.breakdown) == len(dyn_post)
    assert list(report.breakdown['relative_time']) == \
        list(dyn_post['relative_time'])
    assert (report.breakdown['breakdown_M_star'] >= 0).all()


def test_report_overall_matches_simple_aggregation(report):
    assert report.overall['estimate'] == pytest.approx(
        report.simple['att'].iloc[0], rel=1e-6
    )


def test_report_accepts_prefitted_result():
    df = _staggered_panel(seed=7)
    cs = callaway_santanna(df, y='y', g='g', t='t', i='i')
    rpt = cs_report(cs, n_boot=200, random_state=1, verbose=False)
    assert isinstance(rpt, CSReport)
    assert rpt.meta['estimator'] == 'DR'


def test_report_raw_data_requires_column_names():
    df = _staggered_panel(seed=3)
    with pytest.raises(ValueError, match='column names'):
        cs_report(df, n_boot=50, verbose=False)


def test_report_to_text_is_non_empty(report):
    text = report.to_text()
    assert isinstance(text, str)
    assert "Callaway" in text
    assert "Event study" in text
    assert "breakdown" in text.lower()


def test_report_verbose_prints(capsys):
    df = _staggered_panel(seed=9)
    cs_report(df, y='y', g='g', t='t', i='i',
              n_boot=100, random_state=0, verbose=True)
    captured = capsys.readouterr()
    assert "Callaway" in captured.out
    assert "Overall ATT" in captured.out


def test_reproducibility_with_fixed_seed():
    df = _staggered_panel(seed=11)
    a = cs_report(df, y='y', g='g', t='t', i='i',
                  n_boot=200, random_state=42, verbose=False)
    b = cs_report(df, y='y', g='g', t='t', i='i',
                  n_boot=200, random_state=42, verbose=False)
    pd.testing.assert_frame_equal(a.dynamic, b.dynamic)
    pd.testing.assert_frame_equal(a.breakdown, b.breakdown)


# --------------------------------------------------------------------------- #
# Export: to_markdown / to_latex                                              #
# --------------------------------------------------------------------------- #

def test_to_markdown_structure(report):
    md = report.to_markdown()
    assert isinstance(md, str)
    assert md.startswith("## Callaway")
    for section in [
        "Event study",
        "θ(g)",
        "θ(t)",
        "Rambachan",
    ]:
        assert section in md
    # Integer columns should not show .0000 artefacts.
    assert "relative_time" in md
    assert "0.0000" not in md.split("Event study")[1][:500]


def test_to_markdown_float_format(report):
    md3 = report.to_markdown(float_format="%.2f")
    assert "0.07" in md3 or "0.09" in md3
    assert "0.0669" not in md3  # would only appear under %.4f default


def test_to_latex_structure(report):
    tex = report.to_latex(caption='Demo', label='tab:demo')
    assert tex.startswith("\\begin{table}")
    assert "\\caption{Demo}" in tex
    assert "\\label{tab:demo}" in tex
    assert "\\begin{tabular}" in tex
    assert "\\toprule" in tex and "\\midrule" in tex and "\\bottomrule" in tex
    assert tex.rstrip().endswith("\\end{table}")


def test_to_latex_escapes_special_chars(report):
    tex = report.to_latex()
    # Underscores in column names must be escaped.
    assert "ci\\_lower" in tex
    assert "\\chi^2" in tex  # pretrend rendering


def test_to_latex_no_jinja2_required(report):
    """Our booktabs formatter should not depend on jinja2."""
    import sys
    saved = sys.modules.pop('jinja2', None)
    try:
        tex = report.to_latex()
        assert "\\begin{tabular}" in tex
    finally:
        if saved is not None:
            sys.modules['jinja2'] = saved


# --------------------------------------------------------------------------- #
# Plot: 2×2 summary panel                                                     #
# --------------------------------------------------------------------------- #

def test_report_plot_returns_2x2_panel(report):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    fig, axes = report.plot(suptitle="demo")
    assert axes.shape == (2, 2)
    # The four quadrants should each carry a non-empty title.
    titles = [ax.get_title() for ax in axes.ravel()]
    assert all(t for t in titles)
    # Breakdown quadrant has a "Rambachan" in its title.
    assert any("Rambachan" in t for t in titles)


# --------------------------------------------------------------------------- #
# Export: to_excel                                                            #
# --------------------------------------------------------------------------- #

def test_to_excel_roundtrip(report, tmp_path):
    pytest.importorskip("openpyxl")
    out = tmp_path / "cs_report.xlsx"
    returned = report.to_excel(out)
    assert str(returned) == str(out)
    assert out.exists() and out.stat().st_size > 0

    sheets = pd.ExcelFile(out).sheet_names
    assert set(sheets) >= {
        "Summary", "Dynamic", "Group", "Calendar", "Breakdown", "Meta"
    }

    # Dynamic sheet row count matches the in-memory frame.
    dyn_xl = pd.read_excel(out, "Dynamic")
    assert len(dyn_xl) == len(report.dynamic)

    # Summary sheet carries the key header numbers.
    summary_xl = pd.read_excel(out, "Summary")
    keys = set(summary_xl["key"].astype(str))
    assert {"overall_att", "overall_se", "overall_ci_lower",
            "overall_ci_upper", "overall_pvalue"} <= keys


def test_to_excel_breakdown_matches_inmemory(report, tmp_path):
    pytest.importorskip("openpyxl")
    out = tmp_path / "cs_report.xlsx"
    report.to_excel(out)
    bd_xl = pd.read_excel(out, "Breakdown")
    assert len(bd_xl) == len(report.breakdown)
    # relative_time values survive the round-trip.
    assert list(bd_xl["relative_time"]) == list(report.breakdown["relative_time"])


# --------------------------------------------------------------------------- #
# save_to — one-call bundle export                                            #
# --------------------------------------------------------------------------- #

def test_save_to_produces_all_artifacts(tmp_path):
    df = _staggered_panel(seed=13)
    prefix = str(tmp_path / "bundle" / "cs")
    cs_report(df, y='y', g='g', t='t', i='i',
              n_boot=80, random_state=0, verbose=False,
              save_to=prefix)
    assert (tmp_path / "bundle" / "cs.txt").exists()
    assert (tmp_path / "bundle" / "cs.md").exists()
    assert (tmp_path / "bundle" / "cs.tex").exists()
    # Excel + PNG depend on optional deps — tolerate either way.
    try:
        import openpyxl  # noqa: F401
        assert (tmp_path / "bundle" / "cs.xlsx").exists()
    except ImportError:
        pass
    try:
        import matplotlib  # noqa: F401
        assert (tmp_path / "bundle" / "cs.png").exists()
    except ImportError:
        pass


def test_save_to_creates_missing_parent_dirs(tmp_path):
    df = _staggered_panel(seed=14)
    prefix = str(tmp_path / "a" / "b" / "c" / "report")
    cs_report(df, y='y', g='g', t='t', i='i',
              n_boot=50, random_state=0, verbose=False,
              save_to=prefix)
    assert (tmp_path / "a" / "b" / "c" / "report.md").exists()


def test_save_to_files_contain_expected_content(tmp_path):
    df = _staggered_panel(seed=15)
    prefix = str(tmp_path / "cs")
    cs_report(df, y='y', g='g', t='t', i='i',
              n_boot=50, random_state=0, verbose=False,
              save_to=prefix)
    md = (tmp_path / "cs.md").read_text()
    assert "Callaway" in md and "Event study" in md
    tex = (tmp_path / "cs.tex").read_text()
    assert "\\begin{table}" in tex and "\\bottomrule" in tex
    txt = (tmp_path / "cs.txt").read_text()
    assert "Overall ATT" in txt
