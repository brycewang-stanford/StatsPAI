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
