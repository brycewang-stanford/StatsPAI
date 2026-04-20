"""Tests for ``sp.bayes_fuzzy_rd`` — Bayesian fuzzy RD via ratio posterior."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.bayes import BayesianCausalResult, bayes_fuzzy_rd

pymc = pytest.importorskip(
    "pymc",
    reason="PyMC is an optional dependency; skip Bayesian tests if missing.",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fuzzy_rd_dgp(n, pt_treated=0.8, pt_control=0.1, true_late=2.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    side = (x >= 0).astype(int)
    p = np.where(side == 1, pt_treated, pt_control)
    D = rng.binomial(1, p).astype(float)
    Y = 0.5 + 1.0 * x + true_late * D + rng.normal(0, 0.4, n)
    return pd.DataFrame({'y': Y, 'd': D, 'x': x})


@pytest.fixture
def partial_compliance_data():
    return _fuzzy_rd_dgp(800, pt_treated=0.8, pt_control=0.1,
                          true_late=2.0, seed=301)


@pytest.fixture
def full_compliance_data():
    return _fuzzy_rd_dgp(600, pt_treated=1.0, pt_control=0.0,
                          true_late=2.0, seed=302)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_bayes_fuzzy_rd_returns_result(partial_compliance_data):
    r = bayes_fuzzy_rd(partial_compliance_data, y='y', treat='d',
                       running='x', cutoff=0.0, bandwidth=0.5,
                       draws=300, tune=300, chains=2, progressbar=False)
    assert isinstance(r, BayesianCausalResult)
    assert r.estimand == 'LATE'
    assert 'fuzzy RD' in r.method


def test_bayes_fuzzy_rd_top_level_export():
    assert sp.bayes_fuzzy_rd is bayes_fuzzy_rd


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def test_bayes_fuzzy_rd_partial_compliance_recovers_late(partial_compliance_data):
    r = bayes_fuzzy_rd(partial_compliance_data, y='y', treat='d',
                       running='x', cutoff=0.0, bandwidth=0.5,
                       draws=500, tune=500, chains=2, progressbar=False,
                       random_state=17)
    assert r.hdi_lower < 2.0 < r.hdi_upper, (
        f"True LATE=2.0 not covered by 95% HDI "
        f"[{r.hdi_lower:.3f}, {r.hdi_upper:.3f}] "
        f"(mean {r.posterior_mean:.3f})"
    )


def test_bayes_fuzzy_rd_full_compliance_equiv_sharp(full_compliance_data):
    """Under full compliance the fuzzy LATE should concentrate near
    the sharp RD jump."""
    r = bayes_fuzzy_rd(full_compliance_data, y='y', treat='d',
                       running='x', cutoff=0.0, bandwidth=0.5,
                       draws=500, tune=500, chains=2, progressbar=False,
                       random_state=18)
    # Full compliance => first-stage jump ≈ 1 => LATE ≈ ITT_Y
    assert abs(r.posterior_mean - 2.0) < 0.5, (
        f"Full-compliance fuzzy LATE mean {r.posterior_mean:.3f} "
        "diverges too far from sharp truth 2.0"
    )


# ---------------------------------------------------------------------------
# Bandwidth / validation
# ---------------------------------------------------------------------------


def test_bayes_fuzzy_rd_bandwidth_shrinks_sample(partial_compliance_data):
    wide = bayes_fuzzy_rd(partial_compliance_data, y='y', treat='d',
                          running='x', cutoff=0.0, bandwidth=1.0,
                          draws=200, tune=200, chains=2, progressbar=False)
    narrow = bayes_fuzzy_rd(partial_compliance_data, y='y', treat='d',
                            running='x', cutoff=0.0, bandwidth=0.3,
                            draws=200, tune=200, chains=2, progressbar=False)
    assert narrow.n_obs < wide.n_obs


def test_bayes_fuzzy_rd_first_stage_reported(partial_compliance_data):
    r = bayes_fuzzy_rd(partial_compliance_data, y='y', treat='d',
                       running='x', cutoff=0.0, bandwidth=0.5,
                       draws=300, tune=300, chains=2, progressbar=False)
    assert 'first_stage_mean' in r.model_info
    # First-stage jump with 80/10 compliance should be ≈ 0.7
    assert 0.4 < r.model_info['first_stage_mean'] < 1.0


def test_bayes_fuzzy_rd_non_binary_treat_raises(partial_compliance_data):
    df = partial_compliance_data.copy()
    df['d'] = df['d'] * 2.0
    with pytest.raises(ValueError, match='binary'):
        bayes_fuzzy_rd(df, y='y', treat='d', running='x', cutoff=0.0,
                       bandwidth=0.5,
                       draws=50, tune=50, chains=1, progressbar=False)
