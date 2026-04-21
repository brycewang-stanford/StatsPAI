"""Tests for ``sp.bayes_iv(per_instrument=True)`` —
per-instrument LATE with tidy(terms=[...]) multi-row output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.bayes import BayesianCausalResult, BayesianIVResult, bayes_iv

pymc = pytest.importorskip(
    "pymc",
    reason="PyMC is an optional dependency; skip Bayesian tests if missing.",
)


def _two_iv_dgp(n, seed, late=1.5):
    """Two instruments with different first-stage strengths + the
    same structural LATE. Each just-identified sub-fit should
    recover a similar LATE, at different posterior widths
    proportional to each Z's strength.
    """
    rng = np.random.default_rng(seed)
    Z1 = rng.normal(size=n)
    Z2 = rng.normal(size=n)
    U = rng.normal(size=n)
    v = rng.normal(size=n)
    # Z1 strong (π=0.9), Z2 moderate (π=0.4)
    D = 0.9 * Z1 + 0.4 * Z2 + 0.5 * U + v
    Y = late * D + U + rng.normal(size=n)
    return pd.DataFrame({'y': Y, 'd': D, 'z1': Z1, 'z2': Z2})


@pytest.fixture
def two_iv_data():
    return _two_iv_dgp(600, seed=2026, late=1.5)


# ---------------------------------------------------------------------------
# Back-compat: per_instrument=False still behaves as v0.9.15
# ---------------------------------------------------------------------------


def test_no_per_instrument_returns_iv_result(two_iv_data):
    r = bayes_iv(two_iv_data, y='y', treat='d', instrument=['z1', 'z2'],
                 draws=200, tune=200, chains=2, progressbar=False,
                 random_state=0)
    assert isinstance(r, BayesianIVResult)
    assert isinstance(r, BayesianCausalResult)
    # instrument_summaries stays empty by default
    assert r.instrument_summaries == {}
    assert r.instrument_labels == []


def test_no_per_instrument_tidy_single_row(two_iv_data):
    r = bayes_iv(two_iv_data, y='y', treat='d', instrument=['z1', 'z2'],
                 draws=150, tune=150, chains=2, progressbar=False)
    tidy = r.tidy()
    assert len(tidy) == 1
    assert list(tidy['term']) == ['late']


# ---------------------------------------------------------------------------
# Per-instrument fit
# ---------------------------------------------------------------------------


def test_per_instrument_populates_summaries(two_iv_data):
    r = bayes_iv(two_iv_data, y='y', treat='d', instrument=['z1', 'z2'],
                 per_instrument=True,
                 draws=150, tune=150, chains=2, progressbar=False,
                 random_state=3)
    assert set(r.instrument_summaries.keys()) == {'z1', 'z2'}
    assert r.instrument_labels == ['z1', 'z2']
    for name, s in r.instrument_summaries.items():
        assert set(s.keys()) >= {
            'posterior_mean', 'posterior_sd', 'hdi_lower',
            'hdi_upper', 'prob_positive',
        }


def test_per_instrument_tidy_multi_row(two_iv_data):
    r = bayes_iv(two_iv_data, y='y', treat='d', instrument=['z1', 'z2'],
                 per_instrument=True,
                 draws=150, tune=150, chains=2, progressbar=False,
                 random_state=4)
    tidy = r.tidy(terms='per_instrument')
    assert len(tidy) == 2
    assert set(tidy['term']) == {'instrument:z1', 'instrument:z2'}


def test_per_instrument_explicit_list(two_iv_data):
    r = bayes_iv(two_iv_data, y='y', treat='d', instrument=['z1', 'z2'],
                 per_instrument=True,
                 draws=150, tune=150, chains=2, progressbar=False,
                 random_state=5)
    tidy = r.tidy(terms=['late', 'instrument:z1', 'instrument:z2'])
    assert list(tidy['term']) == ['late', 'instrument:z1', 'instrument:z2']


def test_per_instrument_unknown_term_raises(two_iv_data):
    r = bayes_iv(two_iv_data, y='y', treat='d', instrument=['z1', 'z2'],
                 per_instrument=True,
                 draws=80, tune=80, chains=2, progressbar=False,
                 random_state=6)
    with pytest.raises(ValueError, match='Unknown term'):
        r.tidy(terms=['instrument:z99'])


def test_per_instrument_without_fit_raises(two_iv_data):
    r = bayes_iv(two_iv_data, y='y', treat='d', instrument=['z1', 'z2'],
                 per_instrument=False,
                 draws=80, tune=80, chains=2, progressbar=False,
                 random_state=7)
    with pytest.raises(ValueError, match='instrument_summaries'):
        r.tidy(terms='per_instrument')


# ---------------------------------------------------------------------------
# Recovery: each just-identified sub-fit should cover the true LATE=1.5
# ---------------------------------------------------------------------------


def test_per_instrument_covers_true_late(two_iv_data):
    r = bayes_iv(two_iv_data, y='y', treat='d', instrument=['z1', 'z2'],
                 per_instrument=True,
                 draws=400, tune=400, chains=2, progressbar=False,
                 random_state=42)
    for name, s in r.instrument_summaries.items():
        # The credible interval should cover 1.5 on both sub-fits.
        assert s['hdi_lower'] < 1.5 < s['hdi_upper'], (
            f"Z={name}: 95% HDI [{s['hdi_lower']:.3f}, "
            f"{s['hdi_upper']:.3f}] misses LATE=1.5"
        )
