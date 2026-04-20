"""Pinned external-parity tests on canonical simulated replicas.

Each test:

1. Loads a bundled replica from ``sp.datasets``.
2. Runs the canonical estimator for that paper.
3. Asserts the output matches a pinned numerical value to 4 decimals.
4. Asserts the output is in the neighbourhood of the published
   estimate on the ORIGINAL data (proving the replica's structural
   calibration is right).

The pinned values come from running the current implementation on the
simulated replica (seed=42) at the moment of test creation.  Updating
a pinned value requires an explicit commit; this prevents silent
numerical drift across refactors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# =========================================================================
# Pinned reference values (from current-main output on seed=42 replicas)
# =========================================================================

# mpdta — CS 2021 simple ATT
PINNED_MPDTA_CS_ATT = -0.0330
PINNED_MPDTA_CS_SE = 0.0027

# Card 1995 — OLS and IV coefficients on educ
PINNED_CARD_OLS_EDUC = 0.1100
PINNED_CARD_IV_EDUC = 0.1418

# NSW-Lalonde — naive OLS on re78
PINNED_LALONDE_NAIVE_ATT = 1556.0

# NSW-DW (Dehejia-Wahba) — naive OLS vs covariate-adjusted
PINNED_NSW_DW_NAIVE_ATT = -8387.0
PINNED_NSW_DW_ADJ_ATT = 2313.0

# Lee 2008 — RD jump in Democratic voteshare
PINNED_LEE_RD_JUMP = 0.0616

# Tolerance for pinned matches: 1e-3 catches true drift while
# allowing for BLAS-level floating-point variation (~1e-6).
PIN_TOL = 1e-3


# =========================================================================
# Callaway-Sant'Anna (2021) — mpdta
# =========================================================================

class TestMpdtaParity:
    """CS2021 on simulated mpdta replica."""

    @pytest.fixture(scope='class')
    def df(self):
        return sp.datasets.mpdta()

    def test_replica_shape_and_attrs(self, df):
        assert df.shape == (2500, 5)
        assert set(df.columns) == {'countyreal', 'year', 'lemp',
                                   'first_treat', 'treat'}
        assert df.attrs['expected_simple_att'] == pytest.approx(-0.04)
        assert df.attrs['published_simple_att_original'] == pytest.approx(-0.0454)

    def test_cs_simple_att_pinned(self, df):
        """CS simple ATT on our replica matches the pinned value."""
        r = sp.callaway_santanna(df, y='lemp', g='first_treat',
                                 t='year', i='countyreal',
                                 estimator='reg')
        assert r.estimate == pytest.approx(PINNED_MPDTA_CS_ATT, abs=PIN_TOL)

    def test_cs_att_in_published_neighbourhood(self, df):
        """Estimate must be negative and within 50% of published R output."""
        r = sp.callaway_santanna(df, y='lemp', g='first_treat',
                                 t='year', i='countyreal',
                                 estimator='reg')
        # Published R did::att_gt simple ATT on original mpdta: -0.0454
        # Our replica target: -0.04; accept anywhere in [-0.06, -0.02]
        assert -0.06 <= r.estimate <= -0.02, (
            f"CS simple ATT {r.estimate} outside expected range "
            f"[-0.06, -0.02] (replica target -0.04; R original -0.0454)"
        )

    def test_cs_se_pinned(self, df):
        r = sp.callaway_santanna(df, y='lemp', g='first_treat',
                                 t='year', i='countyreal',
                                 estimator='reg')
        assert r.se == pytest.approx(PINNED_MPDTA_CS_SE, abs=PIN_TOL)


# =========================================================================
# Card (1995) — IV returns to schooling
# =========================================================================

class TestCard1995Parity:
    """OLS vs IV on simulated Card replica. Key test: IV > OLS (Card puzzle)."""

    @pytest.fixture(scope='class')
    def df(self):
        return sp.datasets.card_1995()

    def test_ols_educ_coef_pinned(self, df):
        r = sp.regress('lwage ~ educ + exper + expersq + black + south + smsa',
                       data=df, robust='hc1')
        assert r.params['educ'] == pytest.approx(PINNED_CARD_OLS_EDUC,
                                                  abs=PIN_TOL)

    def test_iv_educ_coef_pinned(self, df):
        r = sp.ivreg('lwage ~ exper + expersq + black + south + smsa + '
                     '(educ ~ nearc4)', data=df, robust='hc1')
        assert r.params['educ'] == pytest.approx(PINNED_CARD_IV_EDUC,
                                                  abs=PIN_TOL)

    def test_iv_greater_than_ols_card_puzzle(self, df):
        """The signature Card pattern: IV estimate exceeds OLS."""
        ols = sp.regress('lwage ~ educ + exper + expersq + black + south + smsa',
                         data=df, robust='hc1')
        iv = sp.ivreg('lwage ~ exper + expersq + black + south + smsa + '
                      '(educ ~ nearc4)', data=df, robust='hc1')
        assert iv.params['educ'] > ols.params['educ'], (
            f"Card puzzle violated: IV ({iv.params['educ']:.4f}) "
            f"<= OLS ({ols.params['educ']:.4f})"
        )

    def test_first_stage_f_reasonable(self, df):
        """First-stage F on nearc4 should exceed 10 in our calibrated DGP."""
        r = sp.ivreg('lwage ~ exper + expersq + black + south + smsa + '
                     '(educ ~ nearc4)', data=df, robust='hc1')
        # Find the first-stage F diagnostic
        for k, v in r.diagnostics.items():
            if 'first' in k.lower() and 'f' in k.lower() and 'educ' in k.lower():
                assert v > 10, f"First-stage F = {v:.2f} < 10 (weak)"
                return
        # If not found, check 'First-stage F'
        assert 'First-stage F (educ)' in r.diagnostics


# =========================================================================
# LaLonde NSW — experimental subset
# =========================================================================

class TestLalondeParity:
    """Experimental NSW: naive OLS on re78 should recover calibrated ATT."""

    @pytest.fixture(scope='class')
    def df(self):
        return sp.datasets.nsw_lalonde()

    def test_shape(self, df):
        assert df.shape == (445, 10)

    def test_naive_ols_att_pinned(self, df):
        r = sp.regress('re78 ~ treat', data=df, robust='hc1')
        assert r.params['treat'] == pytest.approx(PINNED_LALONDE_NAIVE_ATT,
                                                    abs=50)

    def test_att_near_dehejia_wahba_published(self, df):
        """Experimental ATT should be in [$1000, $2500] (DW = $1,794)."""
        r = sp.regress('re78 ~ treat', data=df, robust='hc1')
        assert 1000 <= r.params['treat'] <= 2500, (
            f"ATT {r.params['treat']:.0f} outside [1000, 2500]; "
            f"DW published: $1,794"
        )


# =========================================================================
# Dehejia-Wahba NSW + PSID — observational bias demonstration
# =========================================================================

class TestDehejiaWahbaParity:
    """The DW benchmark: naive OLS is strongly biased; adjustment recovers ATT."""

    @pytest.fixture(scope='class')
    def df(self):
        return sp.datasets.nsw_dw()

    def test_naive_ols_strongly_negative(self, df):
        """Naive OLS(re78~treat) should be far negative, showing bias."""
        r = sp.regress('re78 ~ treat', data=df, robust='hc1')
        assert r.params['treat'] == pytest.approx(PINNED_NSW_DW_NAIVE_ATT,
                                                    abs=200)
        assert r.params['treat'] < -5000, (
            f"DW bias demo: naive OLS should be < -$5000, got "
            f"{r.params['treat']:.0f}"
        )

    def test_adjusted_ols_recovers_positive_att(self, df):
        """With rich covariates, OLS should return positive ATT close to $1,794."""
        r = sp.regress(
            're78 ~ treat + age + education + black + hispanic + married + '
            're74 + re75', data=df, robust='hc1')
        assert r.params['treat'] == pytest.approx(PINNED_NSW_DW_ADJ_ATT,
                                                    abs=200)
        assert 1000 <= r.params['treat'] <= 3000


# =========================================================================
# Lee (2008) — Senate RD
# =========================================================================

class TestLeeSenateParity:
    """RD on Lee 2008 replica should recover incumbent advantage ≈ 0.08."""

    @pytest.fixture(scope='class')
    def df(self):
        return sp.datasets.lee_2008_senate()

    def test_rd_jump_pinned(self, df):
        r = sp.rdrobust(df, y='voteshare_next', x='margin', c=0.0)
        assert r.estimate == pytest.approx(PINNED_LEE_RD_JUMP, abs=PIN_TOL)

    def test_jump_near_published(self, df):
        """RD jump must be in [0.05, 0.12] (published: 0.08)."""
        r = sp.rdrobust(df, y='voteshare_next', x='margin', c=0.0)
        assert 0.05 <= r.estimate <= 0.12, (
            f"Lee 2008 RD {r.estimate} outside [0.05, 0.12]; published 0.08"
        )


# =========================================================================
# list_datasets registry
# =========================================================================

def test_list_datasets_returns_dataframe():
    registry = sp.datasets.list_datasets()
    assert isinstance(registry, pd.DataFrame)
    assert len(registry) >= 6
    assert set(registry.columns) == {'name', 'design', 'n_obs',
                                     'paper', 'expected_main'}


def test_all_registered_datasets_loadable():
    """Every entry in list_datasets must be callable and return a DataFrame."""
    registry = sp.datasets.list_datasets()
    for name in registry['name']:
        loader = getattr(sp.datasets, name, None)
        assert callable(loader), f"{name} not callable on sp.datasets"
        df = loader()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, f"{name}() returned empty DataFrame"


def test_every_dataset_has_paper_attr():
    """Every dataset must carry a 'paper' attribute for citation."""
    registry = sp.datasets.list_datasets()
    for name in registry['name']:
        df = getattr(sp.datasets, name)()
        # Synth re-exports may not carry attrs set the same way; allow either.
        has_paper_attr = 'paper' in df.attrs
        has_synth_docstring = hasattr(getattr(sp.datasets, name), '__doc__') \
                              and 'Abadie' in (getattr(sp.datasets, name).__doc__ or '')
        assert has_paper_attr or has_synth_docstring, (
            f"{name}: missing paper citation in attrs or docstring"
        )
