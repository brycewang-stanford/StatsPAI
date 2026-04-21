"""Tests for v0.9.15 BayesianMTEResult.tidy(terms=...) multi-term output."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pymc = pytest.importorskip(
    "pymc",
    reason="PyMC is an optional dependency; skip Bayesian tests if missing.",
)

from statspai.bayes import bayes_mte
from statspai.bayes._base import BayesianMTEResult


def _hv_dgp(n, slope, seed):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=n)
    U_D = rng.uniform(0, 1, size=n)
    p = 1.0 / (1.0 + np.exp(-0.8 * Z))
    D = (p > U_D).astype(float)
    Y = 1.0 + slope * U_D * D + 0.3 * rng.normal(size=n)
    return pd.DataFrame({'y': Y, 'd': D, 'z': Z})


@pytest.fixture
def fit_result():
    df = _hv_dgp(300, slope=1.5, seed=11)
    return bayes_mte(df, y='y', treat='d', instrument='z',
                     mte_method='hv_latent', poly_u=1,
                     draws=200, tune=200, chains=2, progressbar=False,
                     random_state=11)


EXPECTED_COLUMNS = {
    'term', 'estimate', 'std_error', 'statistic', 'p_value',
    'conf_low', 'conf_high', 'prob_positive', 'hdi_prob',
}


# ---------------------------------------------------------------------------
# Back-compat: default path unchanged
# ---------------------------------------------------------------------------


def test_tidy_default_is_single_row_ate(fit_result):
    df = fit_result.tidy()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_tidy_default_schema_unchanged(fit_result):
    df = fit_result.tidy()
    assert set(df.columns) == EXPECTED_COLUMNS


# ---------------------------------------------------------------------------
# Single-term paths
# ---------------------------------------------------------------------------


def test_tidy_ate_single_row(fit_result):
    """Round-B review-H1 fix: ``tidy(terms='ate')`` emits the same
    long label as the parent default (``estimand.lower()``), so a
    user mixing ``tidy()`` and ``tidy(terms='ate')`` calls inside a
    concat pipeline gets uniform ``term`` values."""
    df = fit_result.tidy(terms='ate')
    assert len(df) == 1
    assert df['term'].iloc[0] == fit_result.estimand.lower()


def test_tidy_ate_matches_default_path(fit_result):
    """terms='ate' must be byte-identical to the default tidy() call."""
    default_row = fit_result.tidy().iloc[0].to_dict()
    explicit_row = fit_result.tidy(terms='ate').iloc[0].to_dict()
    # Same dict keys
    assert set(default_row.keys()) == set(explicit_row.keys())
    # Same term label
    assert default_row['term'] == explicit_row['term']
    # Same numeric values (NaN-safe comparison)
    for k in ('estimate', 'std_error', 'conf_low', 'conf_high',
              'prob_positive', 'hdi_prob'):
        a, b = default_row[k], explicit_row[k]
        if np.isnan(a) and np.isnan(b):
            continue
        assert a == b, f"{k}: default={a} vs explicit={b}"


def test_tidy_att_single_row(fit_result):
    df = fit_result.tidy(terms='att')
    assert len(df) == 1
    assert df['term'].iloc[0] == 'att'
    assert np.isfinite(df['estimate'].iloc[0])
    assert np.isfinite(df['std_error'].iloc[0])


def test_tidy_atu_single_row(fit_result):
    df = fit_result.tidy(terms='atu')
    assert df['term'].iloc[0] == 'atu'


# ---------------------------------------------------------------------------
# Multi-row path
# ---------------------------------------------------------------------------


def test_tidy_multi_row_preserves_order(fit_result):
    df = fit_result.tidy(terms=['atu', 'ate', 'att'])
    # Note: under the Round-B fix, the 'ate' request emits the long
    # estimand label (byte-compat with default tidy()). ATT / ATU
    # stay short.
    assert df['term'].tolist() == [
        'atu', fit_result.estimand.lower(), 'att',
    ]


def test_tidy_multi_row_schema_rectangular(fit_result):
    df = fit_result.tidy(terms=['ate', 'att', 'atu'])
    assert len(df) == 3
    assert set(df.columns) == EXPECTED_COLUMNS
    # All rows have finite estimates on a clean fit
    assert np.isfinite(df['estimate']).all()


def test_tidy_concat_workflow(fit_result):
    """Simulate broom pipeline: concat multiple fits' 3-term tables."""
    r1 = fit_result
    df1 = r1.tidy(terms=['ate', 'att'])
    df2 = r1.tidy(terms=['ate', 'att'])  # reuse same fit, testing shape
    concat = pd.concat([df1.assign(fit='r1'),
                        df2.assign(fit='r2')], ignore_index=True)
    assert len(concat) == 4
    assert {'fit'}.issubset(concat.columns)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_tidy_invalid_term_raises(fit_result):
    with pytest.raises(ValueError, match='bogus'):
        fit_result.tidy(terms=['bogus'])


def test_tidy_mixed_valid_invalid_raises(fit_result):
    with pytest.raises(ValueError, match='bogus'):
        fit_result.tidy(terms=['ate', 'bogus', 'att'])


# ---------------------------------------------------------------------------
# Back-compat: NaN prob_positive on old results
# ---------------------------------------------------------------------------


def test_tidy_emits_row_when_prob_positive_nan():
    """A BayesianMTEResult stub where att_prob_positive is NaN
    (simulating a result deserialised from pre-v0.9.15) should still
    emit an ATT row, with `prob_positive=NaN`, not crash."""
    stub = BayesianMTEResult(
        method='stub',
        estimand='ATE',
        posterior_mean=0.5,
        posterior_median=0.5,
        posterior_sd=0.1,
        hdi_lower=0.3,
        hdi_upper=0.7,
        prob_positive=0.99,
        rhat=1.0,
        ess=400.0,
        n_obs=100,
        hdi_prob=0.95,
        model_info={'inference': 'nuts', 'chains': 2, 'draws': 100},
        att=0.6,
        atu=0.4,
        att_sd=0.08,
        att_hdi_lower=0.45,
        att_hdi_upper=0.75,
        # att_prob_positive intentionally left at default NaN
        atu_sd=0.09,
        atu_hdi_lower=0.24,
        atu_hdi_upper=0.56,
    )
    df = stub.tidy(terms=['ate', 'att', 'atu'])
    assert len(df) == 3
    att_row = df[df['term'] == 'att'].iloc[0]
    assert np.isnan(att_row['prob_positive'])
    # Other columns still finite
    assert np.isfinite(att_row['estimate'])
    assert np.isfinite(att_row['std_error'])


# ---------------------------------------------------------------------------
# Prob-positive scalars are populated on a real fit
# ---------------------------------------------------------------------------


def test_att_atu_prob_positive_populated(fit_result):
    assert np.isfinite(fit_result.att_prob_positive)
    assert np.isfinite(fit_result.atu_prob_positive)
    assert 0.0 <= fit_result.att_prob_positive <= 1.0
    assert 0.0 <= fit_result.atu_prob_positive <= 1.0
