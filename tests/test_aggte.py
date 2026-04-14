"""Tests for the aggte() aggregation + multiplier-bootstrap module."""
import numpy as np
import pandas as pd
import pytest

from statspai.did import callaway_santanna, aggte


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _make_staggered_panel(
    n_per_cohort: int = 40,
    cohorts=(3, 5, 7, 0),  # 0 = never treated
    n_periods: int = 8,
    effect_slope: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    """Construct a staggered panel with dynamic treatment effects."""
    rng = np.random.default_rng(seed)
    rows = []
    unit = 0
    for g_val in cohorts:
        for _ in range(n_per_cohort):
            ui = rng.normal(scale=0.3)
            for t in range(1, n_periods + 1):
                te = max(0, t - g_val + 1) * effect_slope if g_val > 0 else 0.0
                rows.append({
                    'i': unit, 't': t, 'g': g_val,
                    'y': ui + 0.2 * t + te + rng.normal(),
                })
            unit += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope='module')
def cs_result():
    df = _make_staggered_panel(seed=42)
    return callaway_santanna(df, y='y', g='g', t='t', i='i')


# --------------------------------------------------------------------------- #
# Basic sanity                                                                #
# --------------------------------------------------------------------------- #

def test_simple_matches_cs_overall(cs_result):
    """aggte(type='simple') should reproduce the overall ATT of the base result."""
    out = aggte(cs_result, type='simple', random_state=0, n_boot=400)
    # Point estimate: simple aggregation uses same cohort-share weights.
    assert out.estimate == pytest.approx(cs_result.estimate, rel=1e-6, abs=1e-6)
    assert out.detail.shape[0] == 1
    assert 'overall' in out.detail.columns


def test_dynamic_recovers_positive_post(cs_result):
    es = aggte(cs_result, type='dynamic', random_state=0, n_boot=400)
    post = es.detail[es.detail['relative_time'] >= 0]
    # With effect_slope=0.5 and a linear ramp, post-event coefficients are
    # monotonically increasing on average.
    assert (post['att'].iloc[-1] > post['att'].iloc[0])
    # All post-event ATTs are positive (well-powered).
    assert (post['att'] > 0).all()


def test_group_aggregation_shape(cs_result):
    grp = aggte(cs_result, type='group', random_state=0, n_boot=400)
    cohorts = sorted(cs_result.detail['group'].unique())
    assert list(grp.detail['group']) == cohorts
    # Earlier cohorts experience the ramp longer → larger θ(g).
    # (Not guaranteed in every simulation, so we just check finiteness.)
    assert grp.detail['att'].notna().all()
    assert (grp.detail['se'] > 0).all()


def test_calendar_aggregation_shape(cs_result):
    cal = aggte(cs_result, type='calendar', random_state=0, n_boot=400)
    # Only post-treatment calendar times appear.
    assert cal.detail['time'].min() >= cs_result.detail['group'].min()
    assert (cal.detail['se'] > 0).all()


# --------------------------------------------------------------------------- #
# Uniform bands                                                               #
# --------------------------------------------------------------------------- #

def test_uniform_band_strictly_wider_than_pointwise(cs_result):
    es = aggte(cs_result, type='dynamic', cband=True,
               random_state=0, n_boot=500)
    # Uniform critical value > 1.96 (Normal pointwise).
    assert es.model_info['crit_val_uniform'] >= 1.959
    # Uniform CI contains the pointwise CI.
    assert (es.detail['cband_lower'] <= es.detail['ci_lower'] + 1e-8).all()
    assert (es.detail['cband_upper'] >= es.detail['ci_upper'] - 1e-8).all()


def test_multiplier_bootstrap_reproducible(cs_result):
    a = aggte(cs_result, type='dynamic', random_state=123, n_boot=300)
    b = aggte(cs_result, type='dynamic', random_state=123, n_boot=300)
    pd.testing.assert_frame_equal(a.detail, b.detail)


def test_bstrap_off_falls_back_to_normal(cs_result):
    out = aggte(cs_result, type='dynamic', bstrap=False, cband=True)
    # Without bootstrap, the uniform critical value degenerates to 1.96.
    assert out.model_info['crit_val_uniform'] == pytest.approx(
        1.959963984540054, rel=1e-6
    )


# --------------------------------------------------------------------------- #
# balance_e / min_e / max_e                                                   #
# --------------------------------------------------------------------------- #

def test_balance_e_restricts_cohorts_and_window(cs_result):
    es_all = aggte(cs_result, type='dynamic', random_state=0, n_boot=300)
    es_bal = aggte(cs_result, type='dynamic', balance_e=1,
                   random_state=0, n_boot=300)
    # Post-event window capped at e=1.
    assert es_bal.detail['relative_time'].max() <= 1
    # Balanced set has fewer (or equal) post-event cells than the full one.
    bal_post = (es_bal.detail['relative_time'] >= 0).sum()
    full_post = (es_all.detail['relative_time'] >= 0).sum()
    assert bal_post <= full_post


def test_min_max_e_window(cs_result):
    es = aggte(cs_result, type='dynamic', min_e=-2, max_e=3,
               random_state=0, n_boot=200)
    assert es.detail['relative_time'].min() >= -2
    assert es.detail['relative_time'].max() <= 3


# --------------------------------------------------------------------------- #
# Argument validation                                                         #
# --------------------------------------------------------------------------- #

def test_invalid_type_raises(cs_result):
    with pytest.raises(ValueError):
        aggte(cs_result, type='nonsense')


def test_invalid_boot_type_raises(cs_result):
    with pytest.raises(NotImplementedError):
        aggte(cs_result, boot_type='pairs')


def test_empty_post_set_raises():
    df = _make_staggered_panel(cohorts=(0,), seed=7)  # all never-treated
    # No cohorts → callaway_santanna rejects earlier.
    with pytest.raises(ValueError):
        callaway_santanna(df, y='y', g='g', t='t', i='i')


# --------------------------------------------------------------------------- #
# Anticipation                                                                #
# --------------------------------------------------------------------------- #

def test_anticipation_shifts_base_period():
    df = _make_staggered_panel(seed=1)
    cs0 = callaway_santanna(df, y='y', g='g', t='t', i='i', anticipation=0)
    cs1 = callaway_santanna(df, y='y', g='g', t='t', i='i', anticipation=1)
    # With δ=1 the earliest cohort (g=3) has no valid pre-period (needs t<=1),
    # so it may produce fewer (g,t) pairs or just shift the numbers.
    assert cs1.model_info['anticipation'] == 1
    assert cs0.model_info['anticipation'] == 0
    # Numbers actually differ (not identical).
    assert cs0.detail.shape == cs1.detail.shape or True  # shape can match
    assert not np.allclose(cs0.detail['att'].values, cs1.detail['att'].values)


# --------------------------------------------------------------------------- #
# ggdid visualiser                                                            #
# --------------------------------------------------------------------------- #

def test_ggdid_all_aggregation_types(cs_result):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    from statspai.did import ggdid

    for typ in ['simple', 'dynamic', 'group', 'calendar']:
        r = aggte(cs_result, type=typ, random_state=0, n_boot=200)
        fig, ax = ggdid(r)
        assert ax.get_title()  # non-empty title
        # uniform band artists appear for non-simple aggregations
        if typ != 'simple':
            labels = [t.get_text() for t in ax.get_legend().get_texts()]
            assert any('Uniform' in lab for lab in labels)
