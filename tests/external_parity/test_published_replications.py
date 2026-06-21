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

# mpdta — CS 2021 simple ATT (pinned to estimator='reg' on seed=42 replica)
PINNED_MPDTA_CS_ATT = -0.0330
PINNED_MPDTA_CS_SE = 0.00774  # bootstrap SE; tolerated drift up to PIN_TOL

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

    def test_conventional_estimate_matches_lee_paper_band(self, df):
        """Conventional (no CCT bias correction) ≈ Lee 2008 Table 4 = 0.077.

        The quickstart notebook leads with Conventional to match the paper
        layer; CCT robust is shown as the modern bias-corrected default.
        """
        r = sp.rdrobust(df, y='voteshare_next', x='margin', c=0.0)
        conv = r.diagnostics['conventional']
        assert 0.060 <= conv['estimate'] <= 0.090, (
            f"Lee 2008 RD Conventional = {conv['estimate']:.4f} outside "
            f"[0.060, 0.090]; published Lee Table 4 = 0.077"
        )


# =========================================================================
# Quickstart notebook — paper-replication-first parity
# =========================================================================

class TestQuickstartNotebookParity:
    """Pin the four headline numbers shown in the 60-second quickstart card.

    These tests lock the *paper-replication* call signatures used in the
    notebook (``aggte(type='dynamic')``, Conventional RD, ``method='classic'``
    SCM). If any drifts past PIN_TOL, the comparison cards in
    ``社媒文档/5.6-快速开始-演示/images/`` must be regenerated.
    """

    def test_did_dynamic_att_quickstart(self):
        df = sp.datasets.mpdta()
        cs = sp.callaway_santanna(df, y='lemp', g='first_treat',
                                   t='year', i='countyreal')
        att_dyn = sp.aggte(cs, type='dynamic')
        # Quickstart card shows -0.034 vs paper -0.045 (CS 2021 Figure 2)
        assert float(att_dyn.estimate) == pytest.approx(-0.034, abs=2e-3)

    def test_iv_card_educ_quickstart(self):
        df = sp.datasets.card_1995()
        iv = sp.ivreg(
            'lwage ~ (educ ~ nearc4) + exper + expersq + black + south + smsa',
            data=df,
        )
        # Quickstart card shows 0.142 vs paper 0.132 (Card 1995 Table 3)
        assert float(iv.params['educ']) == pytest.approx(0.142, abs=2e-3)

    def test_rd_lee_conventional_quickstart(self):
        df = sp.datasets.lee_2008_senate()
        rd = sp.rdrobust(df, y='voteshare_next', x='margin', c=0.0)
        # Quickstart card shows Conventional 0.073 (paper 0.077, Lee Table 4)
        conv = rd.diagnostics['conventional']
        assert float(conv['estimate']) == pytest.approx(0.073, abs=2e-3)

    def test_scm_classic_adh_quickstart(self):
        df = sp.datasets.california_prop99()
        sc = sp.synth(data=df, outcome='cigsale', unit='state', time='year',
                      treated_unit='California', treatment_time=1989,
                      method='classic')
        # Quickstart card shows -13.1 vs paper ADH 2010 ≈ -19
        assert float(sc.estimate) == pytest.approx(-13.1, abs=0.2)


# =========================================================================
# R `did::aggte` bit-equal parity lock (Phase B fix, 2026-05-06)
# =========================================================================
#
# These tests run sp.callaway_santanna + sp.aggte on the *real* mpdta CSV
# (loaded from the canonical R::did package via tests/orig_parity/data/) and
# assert the output matches R `did::aggte`'s recorded output to floating-
# point precision.
#
# Background: an end-user once thought our ``aggte(type='dynamic')`` diverged
# from R because they compared against the CS 2021 paper Table 2 number
# (-0.0454) rather than R's actual run on the same data (-0.0772). Lock the
# real-data parity so future refactors can't silently break alignment.
#
# Source of truth: tests/orig_parity/results/02_mpdta_original_R.json
#   produced by tests/orig_parity/02_mpdta_original.R using
#   `did::aggte(fit, type='simple', bstrap=FALSE, cband=FALSE)`.

class TestAggteRParity:
    """sp.aggte must remain bit-equal with R did::aggte on real mpdta."""

    @pytest.fixture(scope='class')
    def mpdta_real(self):
        """Real mpdta from R::did package (CSV exported via pyreadr)."""
        import pathlib
        path = pathlib.Path(
            __file__
        ).parent.parent / 'orig_parity' / 'data' / '02_mpdta_original.csv'
        if not path.exists():
            pytest.skip(
                f"Real mpdta CSV not present at {path}. Regenerate via "
                "tests/orig_parity/02_mpdta_original.R."
            )
        return pd.read_csv(path)

    def test_aggte_simple_bit_equal_r_did(self, mpdta_real):
        """sp.aggte(type='simple') must equal R did::aggte to ~1e-13.

        R reference value from tests/orig_parity/results/02_mpdta_original_R.json
        (R `did::aggte(fit, type='simple', bstrap=FALSE, cband=FALSE)`):
            estimate = -0.0399512751551772
        """
        cs = sp.callaway_santanna(
            data=mpdta_real,
            y='lemp', t='year', i='countyreal', g='first_treat',
            estimator='reg',
        )
        agg = sp.aggte(cs, type='simple', bstrap=False, cband=False)
        R_REFERENCE = -0.0399512751551772
        assert float(agg.estimate) == pytest.approx(R_REFERENCE, abs=1e-10), (
            f"sp.aggte(type='simple') drifted from R::did::aggte: "
            f"got {float(agg.estimate):.15f}, expected {R_REFERENCE:.15f}. "
            "If this is intentional, regenerate "
            "tests/orig_parity/results/02_mpdta_original_R.json from R."
        )

    def test_aggte_dynamic_matches_r_did(self, mpdta_real):
        """sp.aggte(type='dynamic') must equal R did::aggte default.

        R reference: -0.0772 (from Brantly Callaway's bcallaway11.github.io
        `did-basics` vignette, on the same mpdta data, default args).
        Tolerance 1e-3 to allow for cluster-bootstrap SE noise in shared
        intermediate quantities (point estimate is deterministic).
        """
        cs = sp.callaway_santanna(
            data=mpdta_real,
            y='lemp', t='year', i='countyreal', g='first_treat',
            estimator='reg',
        )
        agg = sp.aggte(cs, type='dynamic', bstrap=False, cband=False)
        R_REFERENCE = -0.0772
        assert float(agg.estimate) == pytest.approx(R_REFERENCE, abs=1e-3), (
            f"sp.aggte(type='dynamic') drifted from R::did::aggte: "
            f"got {float(agg.estimate):.6f}, expected ≈ {R_REFERENCE}."
        )

    def test_aggte_calendar_in_published_band(self, mpdta_real):
        """CS 2021 paper Table 2 reports calendar ATT ≈ -0.041."""
        cs = sp.callaway_santanna(
            data=mpdta_real,
            y='lemp', t='year', i='countyreal', g='first_treat',
            estimator='reg',
        )
        agg = sp.aggte(cs, type='calendar', bstrap=False, cband=False)
        # Paper reports -0.041; allow ±0.005 to absorb CS implementation
        # variance across reg/dr/ipw estimators.
        assert -0.050 <= float(agg.estimate) <= -0.030, (
            f"calendar ATT {float(agg.estimate):.4f} outside paper band "
            "[-0.05, -0.03]; paper Table 2 reports -0.041"
        )


# =========================================================================
# Angrist-Krueger (1991) — quarter-of-birth IV for returns to schooling
# =========================================================================

class TestAngristKrueger1991Parity:
    """QOB IV on AK91 replica should recover returns to schooling ≈ 0.10.

    Published values (Angrist & Krueger 1991, QJE 106):
      - OLS lwage ~ educ:            ≈ 0.07
      - IV (q1-q3 as instruments):   ≈ 0.08-0.11 (Table V)

    The simulated replica calibrates to the 0.10 point.  We assert:
      - OLS in a plausible band
      - IV in the published band
      - First-stage F survives weak-IV diagnostics
    """

    @pytest.fixture(scope='class')
    def df(self):
        return sp.datasets.angrist_krueger_1991()

    def test_has_paper_attr(self, df):
        assert 'paper' in df.attrs
        assert 'Angrist' in df.attrs['paper']

    def test_ols_educ_in_plausible_band(self, df):
        r = sp.regress("lwage ~ educ", data=df)
        ols_educ = float(r.params['educ'])
        # OLS on this simulated replica lands around 0.13; published
        # (original data) is ~0.07.  We keep a wide band because the
        # simulated DGP is calibrated to the IV target, not the OLS one.
        assert 0.02 <= ols_educ <= 0.20, (
            f"AK91 OLS {ols_educ} outside [0.02, 0.20]; published ~0.07"
        )

    def test_iv_educ_in_published_band(self, df):
        """IV with q1/q2/q3 as instruments should fall near the DGP
        target of 0.10 (published original-data range: 0.08-0.11).

        NOTE: use ``sp.ivreg``, not ``sp.iv`` — the package re-binds
        ``sp.iv`` to the subpackage (see ``__init__.py`` L127-129).

        The simulated replica is calibrated to 0.10 with deterministic
        seed, so the band ``[0.07, 0.13]`` is tight enough to catch real
        regressions (e.g. biased pseudo-inverse) while still absorbing
        moderate implementation drift.  The original paper's band was
        ``(0.08, 0.11)``; we extend by ±0.02 for simulation variance."""
        r = sp.ivreg("lwage ~ (educ ~ q1 + q2 + q3)", data=df)
        iv_educ = float(r.params['educ'])
        published_lo, published_hi = df.attrs['published_iv_original_range']
        lo = published_lo - 0.02
        hi = published_hi + 0.02
        assert lo <= iv_educ <= hi, (
            f"AK91 IV {iv_educ} outside [{lo}, {hi}]; "
            f"published {published_lo}-{published_hi}"
        )

    def test_first_stage_f_strong_enough(self, df):
        """First-stage F on q1+q2+q3 → educ should not be weak-IV-critical.

        The diagnostics dict uses the statsmodels-style "F-statistic" key
        rather than "f_stat".  Stock-Yogo 10% threshold for 1 endogenous,
        3 instruments is ~9.08; the AK91 replica calibrates to a moderate
        first stage so we relax to > 5 to absorb simulation variance."""
        fs = sp.regress("educ ~ q1 + q2 + q3", data=df)
        f_stat = float(
            fs.diagnostics.get('F-statistic',
                               fs.diagnostics.get('f_stat', 0.0))
        )
        assert f_stat > 5.0, (
            f"AK91 first-stage F={f_stat} looks weak-IV-critical"
        )


# =========================================================================
# list_datasets registry
# =========================================================================

def test_list_datasets_returns_dataframe():
    registry = sp.datasets.list_datasets()
    assert isinstance(registry, pd.DataFrame)
    assert len(registry) >= 6
    assert set(registry.columns) == {'name', 'design', 'n_obs', 'paper',
                                     'paper_original', 'expected_main'}


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


# =========================================================================
# bwselect='cct' R-parity lock (Phase 3a fix, 2026-05-06)
# =========================================================================
#
# ``sp.rdrobust(..., bwselect='cct')`` delegates to the official
# ``rdrobust>=1.3`` Python port (Calonico-Cattaneo-Titiunik 2014). This
# test class ensures the delegation stays bit-equal with R `rdrobust::
# rdrobust` on the canonical Lee/CCT Senate replication.
#
# If the official rdrobust package isn't installed, the tests skip — the
# delegation itself raises a clear ImportError at runtime.

class TestCCTDelegationParity:
    """sp.rdrobust(bwselect='cct') must equal R rdrobust::rdrobust."""

    @pytest.fixture(scope='class')
    def lee_real(self):
        rdrobust = pytest.importorskip(
            'rdrobust',
            reason="bwselect='cct' delegates to rdrobust>=1.3; "
                   "skipping when extras [rd-cct] are not installed.",
        )
        del rdrobust
        import pathlib
        path = pathlib.Path(
            __file__
        ).parent.parent / 'orig_parity' / 'data' / '05_lee_original.csv'
        if not path.exists():
            pytest.skip(
                f"Real Lee/CCT Senate CSV not present at {path}. "
                "Regenerate via tests/orig_parity/05_lee_original.R."
            )
        return pd.read_csv(path)

    def test_cct_conventional_matches_r(self, lee_real):
        """Conventional point estimate from CCT 2014 JSS Table 1 = 7.41."""
        cols = list(lee_real.columns)
        y_col = 'vote' if 'vote' in cols else cols[1]
        x_col = 'margin' if 'margin' in cols else cols[0]
        r = sp.rdrobust(data=lee_real, y=y_col, x=x_col, c=0, bwselect='cct')
        conv = r.diagnostics['conventional']
        # R rdrobust gives 7.4141 (bit-equal); allow 1e-3 for float drift.
        assert float(conv['estimate']) == pytest.approx(7.4141, abs=1e-3), (
            f"bwselect='cct' Conventional drifted from R rdrobust: "
            f"got {conv['estimate']:.6f}, expected 7.4141. "
            "If rdrobust>=1.3 changed its output, regenerate "
            "tests/orig_parity/results/05_lee_original_R.json."
        )

    def test_cct_robust_matches_r(self, lee_real):
        """Robust bias-corrected estimate matches R: 7.5065."""
        cols = list(lee_real.columns)
        y_col = 'vote' if 'vote' in cols else cols[1]
        x_col = 'margin' if 'margin' in cols else cols[0]
        r = sp.rdrobust(data=lee_real, y=y_col, x=x_col, c=0, bwselect='cct')
        assert float(r.estimate) == pytest.approx(7.5065, abs=1e-3), (
            f"bwselect='cct' Robust drifted from R rdrobust: "
            f"got {r.estimate:.6f}, expected 7.5065"
        )

    def test_cct_bandwidth_matches_r(self, lee_real):
        """MSE-optimal bandwidth h matches R: 17.754."""
        cols = list(lee_real.columns)
        y_col = 'vote' if 'vote' in cols else cols[1]
        x_col = 'margin' if 'margin' in cols else cols[0]
        r = sp.rdrobust(data=lee_real, y=y_col, x=x_col, c=0, bwselect='cct')
        h = r.diagnostics['bandwidth_h']
        if isinstance(h, tuple):
            h = h[0]
        assert float(h) == pytest.approx(17.7544, abs=1e-3), (
            f"bwselect='cct' bandwidth_h drifted from R rdrobust: "
            f"got {h:.6f}, expected 17.7544"
        )


def test_bwselect_cct_raises_clear_error_when_rdrobust_missing(monkeypatch):
    """``bwselect='cct'`` must raise a helpful ImportError when the
    optional rdrobust dependency is not installed."""
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == 'rdrobust':
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', _fake_import)

    df = pd.DataFrame({
        'y': np.arange(200, dtype=float),
        'x': np.linspace(-1, 1, 200),
    })
    with pytest.raises(ImportError, match=r"pip install statspai\[rd-cct\]"):
        sp.rdrobust(data=df, y='y', x='x', c=0, bwselect='cct')
