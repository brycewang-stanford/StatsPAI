"""Tests for the repeated-cross-sections (panel=False) path of
callaway_santanna()."""
import numpy as np
import pandas as pd
import pytest

from statspai.did import aggte, callaway_santanna, cs_report, honest_did


def _rcs_panel(
    n_obs: int = 1200,
    cohorts=(3, 5, 7, 0),
    cohort_probs=(0.2, 0.2, 0.2, 0.4),
    n_periods: int = 8,
    effect_slope: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    """Repeated cross-sections: each 'obs' appears at exactly one time."""
    rng = np.random.default_rng(seed)
    rows = []
    for obs in range(n_obs):
        g = int(rng.choice(cohorts, p=cohort_probs))
        t = int(rng.integers(1, n_periods + 1))
        ui = rng.normal(scale=0.3)
        te = max(0, t - g + 1) * effect_slope if g > 0 else 0.0
        y = ui + 0.2 * t + te + rng.normal()
        rows.append({'obs': obs, 't': t, 'g': g, 'y': y})
    return pd.DataFrame(rows)


@pytest.fixture(scope='module')
def rcs_result():
    df = _rcs_panel(seed=42)
    return callaway_santanna(
        df, y='y', g='g', t='t', i='obs',
        estimator='reg', panel=False,
    )


# --------------------------------------------------------------------------- #
# Basic RCS behaviour                                                         #
# --------------------------------------------------------------------------- #

def test_rcs_returns_causal_result(rcs_result):
    from statspai.core.results import CausalResult
    assert isinstance(rcs_result, CausalResult)
    assert rcs_result.model_info['panel'] is False
    assert 'RCS' in rcs_result.model_info['estimator']


def test_rcs_recovers_positive_att(rcs_result):
    assert rcs_result.estimate > 0
    # With n=1200 and effect_slope=0.5 the effect should be significant.
    assert rcs_result.pvalue < 0.01


def test_rcs_produces_full_detail_grid(rcs_result):
    # 3 cohorts × (8 - 1) non-base periods each = 21 (g, t) pairs
    assert len(rcs_result.detail) == 21
    assert {'group', 'time', 'att', 'se',
            'ci_lower', 'ci_upper', 'pvalue',
            'relative_time'} <= set(rcs_result.detail.columns)


def test_rcs_influence_functions_are_obs_level(rcs_result):
    inf = rcs_result._influence_funcs
    assert inf is not None
    # One row per observation, one column per (g, t) pair.
    assert inf.shape[0] == rcs_result.n_obs
    assert inf.shape[1] == len(rcs_result.detail)


# --------------------------------------------------------------------------- #
# aggte / cs_report integration                                               #
# --------------------------------------------------------------------------- #

def test_aggte_dynamic_works_on_rcs(rcs_result):
    es = aggte(rcs_result, type='dynamic', n_boot=200, random_state=0)
    assert {'cband_lower', 'cband_upper'} <= set(es.detail.columns)
    post = es.detail[es.detail['relative_time'] >= 0]
    # Post-event ATTs should be positive (true DGP effect).
    assert (post['att'] > 0).all()


def test_cs_report_accepts_rcs_result(rcs_result):
    rpt = cs_report(rcs_result, n_boot=200, random_state=0, verbose=False)
    assert rpt.overall['estimate'] == pytest.approx(
        rcs_result.estimate, rel=1e-6,
    )
    assert rpt.meta['estimator'] == 'REG (RCS)'


def test_honest_did_on_rcs_result(rcs_result):
    """RCS result's event study should feed into R-R sensitivity."""
    s = honest_did(rcs_result, e=1)
    assert len(s) > 0
    assert (s['ci_upper'] >= s['ci_lower']).all()


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #

def test_rcs_accepts_covariates():
    """Covariates are now supported in RCS mode via residualisation."""
    df = _rcs_panel(seed=1)
    df['x1'] = 0.0  # constant covariate — residualisation should no-op
    r = callaway_santanna(
        df, y='y', g='g', t='t', i='obs',
        x=['x1'], panel=False, estimator='reg',
    )
    # Constant covariate → residualisation leaves Y unchanged up to numerics.
    assert r.model_info['covariates'] == ['x1']


def test_rcs_rejects_non_reg_estimator():
    df = _rcs_panel(seed=2)
    with pytest.raises(NotImplementedError, match="estimator='reg'"):
        callaway_santanna(
            df, y='y', g='g', t='t', i='obs',
            estimator='dr', panel=False,
        )


def test_rcs_rejects_notyettreated_control():
    df = _rcs_panel(seed=3)
    with pytest.raises(NotImplementedError, match='nevertreated'):
        callaway_santanna(
            df, y='y', g='g', t='t', i='obs',
            estimator='reg', control_group='notyettreated', panel=False,
        )


# --------------------------------------------------------------------------- #
# Comparison with panel mode                                                  #
# --------------------------------------------------------------------------- #

def test_rcs_and_panel_agree_when_balanced():
    """On a balanced panel, panel=False and panel=True should produce
    similar overall ATTs (identical up to the different SE formula).

    Point estimates coincide because when every unit is observed in
    both t and base, the cell means equal the unit-level means and
    ΔȲ = Ȳ_t − Ȳ_base, so the 2×2 cell DID equals the first-differences
    DID.
    """
    # Balanced panel: each unit observed in every period.
    rng = np.random.default_rng(7)
    rows = []
    for u in range(200):
        g = [3, 5, 7, 0][u // 50]
        ui = rng.normal(scale=0.3)
        for t in range(1, 9):
            te = max(0, t - g + 1) * 0.5 if g > 0 else 0
            rows.append({'i': u, 't': t, 'g': g,
                         'y': ui + 0.2 * t + te + rng.normal()})
    df = pd.DataFrame(rows)

    cs_panel = callaway_santanna(df, y='y', g='g', t='t', i='i',
                                 estimator='reg')
    cs_rcs = callaway_santanna(df, y='y', g='g', t='t', i='i',
                               estimator='reg', panel=False)
    # Point estimates should coincide at each (g, t).
    merged = cs_panel.detail.merge(
        cs_rcs.detail, on=['group', 'time'], suffixes=('_p', '_r'),
    )
    np.testing.assert_allclose(
        merged['att_p'], merged['att_r'], atol=1e-10,
    )


# --------------------------------------------------------------------------- #
# RCS with covariates (regression adjustment)                                 #
# --------------------------------------------------------------------------- #

def _rcs_panel_with_covariate(n_obs=2000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for obs in range(n_obs):
        g = int(rng.choice([3, 5, 7, 0], p=[0.2, 0.2, 0.2, 0.4]))
        t = int(rng.integers(1, 9))
        xx = rng.normal()
        ui = rng.normal(scale=0.3)
        te = max(0, t - g + 1) * 0.5 if g > 0 else 0
        y = ui + 0.2 * t + 0.8 * xx + te + rng.normal()
        rows.append({'obs': obs, 't': t, 'g': g, 'x1': xx, 'y': y})
    return pd.DataFrame(rows)


def test_rcs_with_covariate_runs():
    df = _rcs_panel_with_covariate(seed=7)
    r = callaway_santanna(df, y='y', g='g', t='t', i='obs',
                          x=['x1'], panel=False, estimator='reg')
    assert r.model_info['panel'] is False
    assert 'covariates' in r.model_info['estimator']
    assert r.model_info['covariates'] == ['x1']
    assert r.estimate > 0 and r.pvalue < 0.01


def test_rcs_covariate_reduces_se():
    """Adding a predictive covariate should narrow the SE (variance
    reduction from residualisation)."""
    df = _rcs_panel_with_covariate(seed=11, n_obs=3000)
    r_no = callaway_santanna(df, y='y', g='g', t='t', i='obs',
                             panel=False, estimator='reg')
    r_cov = callaway_santanna(df, y='y', g='g', t='t', i='obs',
                              x=['x1'], panel=False, estimator='reg')
    assert r_cov.se < r_no.se


def test_rcs_covariate_downstream_aggte_works():
    from statspai.did import aggte
    df = _rcs_panel_with_covariate(seed=3)
    r = callaway_santanna(df, y='y', g='g', t='t', i='obs',
                          x=['x1'], panel=False, estimator='reg')
    es = aggte(r, type='dynamic', n_boot=200, random_state=0)
    post = es.detail[es.detail['relative_time'] >= 0]
    assert (post['att'] > 0).all()
