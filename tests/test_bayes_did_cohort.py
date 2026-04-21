"""Tests for ``sp.bayes_did(cohort=...)`` — staggered-DID ATT-by-cohort
with tidy(terms=[...]) multi-row output.

Under the v0.9.16 extension ``bayes_did`` fits a vector ``tau_cohort``
of length ``n_cohorts`` when the user supplies a ``cohort`` column,
and the resulting ``BayesianDIDResult`` exposes per-cohort summaries
via ``tidy(terms='per_cohort')``. The top-level scalar ATT is the
treated-size-weighted mean of the per-cohort ``τ``'s.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.bayes import (
    BayesianCausalResult,
    BayesianDIDResult,
    bayes_did,
)

pymc = pytest.importorskip(
    "pymc",
    reason="PyMC is an optional dependency; skip Bayesian tests if missing.",
)


@pytest.fixture
def staggered_did_data():
    """Two cohorts with heterogeneous ATTs (2.0 vs 0.5) + control pool."""
    rng = np.random.default_rng(2026)
    rows = []
    for u in range(90):
        if u < 30:
            cohort = 2019    # treated from t=2 onwards, true τ = 2.0
            treat = 1
            post_cutoff = 2
            att = 2.0
        elif u < 60:
            cohort = 2020    # treated from t=3 onwards, true τ = 0.5
            treat = 1
            post_cutoff = 3
            att = 0.5
        else:
            cohort = -1      # never-treated control
            treat = 0
            post_cutoff = 10
            att = 0.0
        alpha_u = rng.normal(0, 0.6)
        for t in range(4):
            post = int(t >= post_cutoff)
            did = int(treat and post)
            y = 1.0 + alpha_u + 0.3 * post + att * did + rng.normal(0, 0.4)
            rows.append({
                'y': y, 'treat': treat, 'post': post,
                'unit': u, 'time': t, 'cohort': cohort,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Back-compat: no `cohort` supplied behaves exactly as v0.9.15
# ---------------------------------------------------------------------------


def test_no_cohort_returns_bayesian_did_result(staggered_did_data):
    r = bayes_did(staggered_did_data, y='y', treat='treat', post='post',
                  draws=200, tune=200, chains=2, progressbar=False,
                  random_state=0)
    assert isinstance(r, BayesianDIDResult)
    assert isinstance(r, BayesianCausalResult)   # subclass relationship
    # cohort_summaries must stay empty on the back-compat path.
    assert r.cohort_summaries == {}
    assert r.cohort_labels == []


def test_no_cohort_tidy_single_row(staggered_did_data):
    r = bayes_did(staggered_did_data, y='y', treat='treat', post='post',
                  draws=150, tune=150, chains=2, progressbar=False)
    tidy = r.tidy()
    assert len(tidy) == 1
    assert list(tidy['term']) == ['att']


# ---------------------------------------------------------------------------
# Per-cohort fit + tidy(terms=['per_cohort'])
# ---------------------------------------------------------------------------


def test_cohort_fit_populates_summaries(staggered_did_data):
    r = bayes_did(staggered_did_data, y='y', treat='treat', post='post',
                  cohort='cohort',
                  draws=300, tune=300, chains=2, progressbar=False,
                  random_state=11)
    assert isinstance(r, BayesianDIDResult)
    # Three cohorts: 2019, 2020, -1 (never-treated control sentinel)
    assert len(r.cohort_labels) == 3
    assert len(r.cohort_summaries) == 3
    # Each summary carries the standard posterior-summary keys
    for lab, s in r.cohort_summaries.items():
        assert set(s.keys()) >= {
            'posterior_mean', 'posterior_sd', 'hdi_lower',
            'hdi_upper', 'prob_positive',
        }


def test_cohort_tidy_per_cohort_multi_row(staggered_did_data):
    r = bayes_did(staggered_did_data, y='y', treat='treat', post='post',
                  cohort='cohort',
                  draws=300, tune=300, chains=2, progressbar=False,
                  random_state=12)
    tidy = r.tidy(terms='per_cohort')
    assert len(tidy) == 3
    # All rows use the 'cohort:<label>' prefix so downstream concat
    # doesn't collide with other estimators' 'att' / 'late' terms.
    assert all(t.startswith('cohort:') for t in tidy['term'])


def test_cohort_tidy_explicit_list(staggered_did_data):
    r = bayes_did(staggered_did_data, y='y', treat='treat', post='post',
                  cohort='cohort',
                  draws=150, tune=150, chains=2, progressbar=False,
                  random_state=13)
    # Explicit: average ATT + two cohort labels in a chosen order.
    tidy = r.tidy(terms=['att', 'cohort:2019', 'cohort:2020'])
    assert list(tidy['term']) == ['att', 'cohort:2019', 'cohort:2020']
    # Non-ATT columns must match the schema for downstream concat.
    assert set(tidy.columns) >= {
        'term', 'estimate', 'std_error', 'conf_low',
        'conf_high', 'prob_positive',
    }


def test_cohort_unknown_term_raises(staggered_did_data):
    r = bayes_did(staggered_did_data, y='y', treat='treat', post='post',
                  cohort='cohort',
                  draws=100, tune=100, chains=2, progressbar=False,
                  random_state=14)
    with pytest.raises(ValueError, match='Unknown term'):
        r.tidy(terms=['cohort:2099'])


def test_per_cohort_without_cohort_fit_raises(staggered_did_data):
    r = bayes_did(staggered_did_data, y='y', treat='treat', post='post',
                  draws=100, tune=100, chains=2, progressbar=False,
                  random_state=15)
    with pytest.raises(ValueError, match='cohort_summaries'):
        r.tidy(terms='per_cohort')


# ---------------------------------------------------------------------------
# Recovery: the 2019 cohort should have a higher ATT than the 2020
# ---------------------------------------------------------------------------


def test_cohort_recovers_ordering(staggered_did_data):
    """2019 cohort τ = 2.0, 2020 cohort τ = 0.5. The posterior means
    must preserve that ordering at n≈270 treated rows."""
    r = bayes_did(staggered_did_data, y='y', treat='treat', post='post',
                  cohort='cohort',
                  draws=400, tune=400, chains=2, progressbar=False,
                  random_state=42)
    m_2019 = r.cohort_summaries['2019']['posterior_mean']
    m_2020 = r.cohort_summaries['2020']['posterior_mean']
    assert m_2019 > m_2020, (
        f"Expected τ_2019 ({m_2019:.3f}) > τ_2020 ({m_2020:.3f}) "
        f"given DGP ATTs of 2.0 vs 0.5"
    )


def test_cohort_weight_recorded_in_model_info(staggered_did_data):
    r = bayes_did(staggered_did_data, y='y', treat='treat', post='post',
                  cohort='cohort',
                  draws=100, tune=100, chains=2, progressbar=False)
    w = r.model_info['cohort_weights']
    assert len(w) == 3
    # Weights are non-negative and sum to 1 (size-weighted average over
    # treated cohorts; control cohort gets weight 0).
    assert all(wi >= 0 for wi in w)
    assert abs(sum(w) - 1.0) < 1e-9
