"""
Numerical validation: StatsPAI vs statsmodels / linearmodels (Stata/R proxies).

These tests verify that StatsPAI's core estimators produce numerically
identical results (to 4+ decimal places) to established backends. Since
StatsPAI wraps statsmodels / linearmodels internally, exact agreement is
expected — these tests guard against regressions from API-layer
transformations, formula parsing, or output formatting that silently
corrupt values.

Each test uses a fixed random seed and a known DGP with analytically
predictable parameters, so failures pinpoint coefficient-level deviations.

Benchmark references:
  - OLS / robust SE: statsmodels OLS (matches Stata `reg, r` and R `lm()`)
  - IV / 2SLS: linearmodels IV2SLS (matches Stata `ivregress 2sls`)
  - Panel FE: linearmodels PanelOLS (matches Stata `xtreg, fe`)
  - DID: known DGP with exact ATT = 3.0
  - RD: known DGP with exact jump = 2.0
  - Matching: known DGP with true ATT = 2.0
"""

import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, 'src')
import statspai as sp


# =====================================================================
# Helpers
# =====================================================================

def _ols_data(seed=42, n=10000):
    """DGP: y = 2 + 3*x1 - 1.5*x2 + N(0, 0.25)."""
    np.random.seed(seed)
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    e = np.random.randn(n) * 0.5
    y = 2.0 + 3.0 * x1 - 1.5 * x2 + e
    return pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})


def _iv_data(seed=1, n=5000):
    """DGP: y = 1 + 2.5*x + u, x = 0.8*z + 0.3*u + noise."""
    np.random.seed(seed)
    z = np.random.randn(n)
    u = np.random.randn(n)
    x = 0.8 * z + 0.3 * u + 0.2 * np.random.randn(n)
    y = 1.0 + 2.5 * x + u + 0.3 * np.random.randn(n)
    return pd.DataFrame({'y': y, 'x': x, 'z': z})


def _panel_data(seed=2, n_id=200, n_t=10):
    """DGP: y = 1.5*x + fe_i + N(0, 0.09)."""
    np.random.seed(seed)
    n_obs = n_id * n_t
    fe = np.repeat(np.random.randn(n_id), n_t)
    x = np.random.randn(n_obs)
    y = 1.5 * x + fe + np.random.randn(n_obs) * 0.3
    return pd.DataFrame({
        'id': np.repeat(range(n_id), n_t),
        'time': np.tile(range(n_t), n_id),
        'x': x, 'y': y,
    })


def _did_data(seed=0, n=2000):
    """DGP: ATT = 3.0 exactly."""
    np.random.seed(seed)
    n_units = n // 4
    df = pd.DataFrame({
        'id': np.repeat(range(n_units), 4),
        'time': np.tile(range(4), n_units),
        'treat': np.repeat(np.random.binomial(1, 0.5, n_units), 4),
    })
    df['post'] = (df['time'] >= 2).astype(int)
    df['y'] = (1 + 2 * df['treat'] + 0.5 * df['time']
               + 3.0 * df['treat'] * df['post']
               + np.random.randn(n) * 0.5)
    return df


def _rd_data(seed=3, n=2000):
    """DGP: y = 0.5*x + 2.0*1(x>=0) + N(0, 0.09)."""
    np.random.seed(seed)
    x = np.random.uniform(-1, 1, n)
    treat = (x >= 0).astype(float)
    y = 0.5 * x + 2.0 * treat + np.random.randn(n) * 0.3
    return pd.DataFrame({'y': y, 'x': x})


def _match_data(seed=4, n=1000):
    """DGP: ATT = 2.0, confounded by x."""
    np.random.seed(seed)
    x = np.random.randn(n)
    treat = (np.random.randn(n) + 0.5 * x > 0).astype(int)
    y = 2.0 * treat + 1.0 * x + np.random.randn(n) * 0.5
    return pd.DataFrame({'y': y, 'treat': treat, 'x': x})


# =====================================================================
# 1. OLS — coefficients, SE, robust SE
# =====================================================================

class TestOLSValidation:
    """Validate sp.regress() against statsmodels OLS."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = _ols_data()
        self.r = sp.regress('y ~ x1 + x2', data=self.df)

    def test_ols_coefficients(self):
        """Coefficients must match DGP within sampling error."""
        assert self.r.params['Intercept'] == pytest.approx(2.0, abs=0.05)
        assert self.r.params['x1'] == pytest.approx(3.0, abs=0.05)
        assert self.r.params['x2'] == pytest.approx(-1.5, abs=0.05)

    def test_ols_exact_vs_statsmodels(self):
        """Exact match to statsmodels (same backend)."""
        import statsmodels.api as sm
        r_sm = sm.OLS.from_formula('y ~ x1 + x2', self.df).fit()
        for var in ['Intercept', 'x1', 'x2']:
            assert self.r.params[var] == pytest.approx(
                r_sm.params[var], rel=1e-10)
            assert self.r.std_errors[var] == pytest.approx(
                r_sm.bse[var], rel=1e-10)

    def test_ols_robust_se(self):
        """HC1 robust SE matches statsmodels."""
        import statsmodels.api as sm
        r_hc1 = sp.regress('y ~ x1 + x2', data=self.df, robust='hc1')
        r_sm = sm.OLS.from_formula('y ~ x1 + x2', self.df).fit(
            cov_type='HC1')
        for var in ['Intercept', 'x1', 'x2']:
            assert r_hc1.std_errors[var] == pytest.approx(
                r_sm.bse[var], rel=1e-8)

    def test_ols_r_squared(self):
        """R-squared must be high for this DGP (true R2 ~ 0.97)."""
        r2 = self.r.diagnostics['R-squared']
        assert 0.95 < r2 < 1.0

    def test_ols_pvalues_significant(self):
        """All true coefficients should be significant at 1%."""
        for i, tv in enumerate(self.r.tvalues):
            assert abs(tv) > 2.576  # z > 2.576 → p < 0.01


# =====================================================================
# 2. IV / 2SLS — coefficients, SE
# =====================================================================

class TestIVValidation:
    """Validate sp.ivreg() against linearmodels IV2SLS."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = _iv_data()
        self.r = sp.ivreg('y ~ (x ~ z)', data=self.df)

    def test_iv_coefficients(self):
        """IV coefficients must recover true params."""
        assert self.r.params['Intercept'] == pytest.approx(1.0, abs=0.1)
        assert self.r.params['x'] == pytest.approx(2.5, abs=0.1)

    def test_iv_exact_vs_linearmodels(self):
        """Exact match to linearmodels IV2SLS."""
        from linearmodels.iv import IV2SLS
        df = self.df.copy()
        df['const'] = 1
        r_lm = IV2SLS(df['y'], df[['const']], df[['x']], df[['z']]).fit()
        assert self.r.params['Intercept'] == pytest.approx(
            r_lm.params['const'], rel=1e-6)
        assert self.r.params['x'] == pytest.approx(
            r_lm.params['x'], rel=1e-6)

    def test_iv_se_vs_linearmodels(self):
        """Standard errors close to linearmodels (small df correction diff)."""
        from linearmodels.iv import IV2SLS
        df = self.df.copy()
        df['const'] = 1
        r_lm = IV2SLS(df['y'], df[['const']], df[['x']], df[['z']]).fit()
        # Allow rel=1e-3 for minor df-correction differences
        assert self.r.std_errors['x'] == pytest.approx(
            r_lm.std_errors['x'], rel=1e-3)

    def test_iv_first_stage_f(self):
        """First-stage F should be large (strong instrument)."""
        # Key may be 'First-stage F' or 'First-stage F (x)' depending
        # on the number of endogenous variables.
        f_keys = [k for k in self.r.diagnostics
                  if 'First-stage F' in k and 'p-value' not in k]
        assert len(f_keys) > 0, (
            f"No first-stage F in diagnostics: {list(self.r.diagnostics)}")
        f_stat = self.r.diagnostics[f_keys[0]]
        assert f_stat > 10  # rule of thumb


# =====================================================================
# 3. Panel FE — coefficients, SE
# =====================================================================

class TestPanelValidation:
    """Validate sp.panel() against linearmodels PanelOLS."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = _panel_data()
        self.r = sp.panel(self.df, 'y ~ x', entity='id', time='time',
                          method='fe')

    def test_panel_fe_coefficient(self):
        """FE coefficient must recover true beta = 1.5."""
        assert self.r.params['x'] == pytest.approx(1.5, abs=0.05)

    def test_panel_fe_exact_vs_linearmodels(self):
        """Exact match to linearmodels PanelOLS."""
        from linearmodels.panel import PanelOLS
        df = self.df.copy().set_index(['id', 'time'])
        r_lm = PanelOLS.from_formula('y ~ x + EntityEffects', df).fit()
        assert self.r.params['x'] == pytest.approx(
            r_lm.params['x'], rel=1e-10)
        assert self.r.std_errors['x'] == pytest.approx(
            r_lm.std_errors['x'], rel=1e-8)


# =====================================================================
# 4. DID — ATT recovery
# =====================================================================

class TestDIDValidation:
    """Validate sp.did() recovers known ATT."""

    def test_did_att_recovery(self):
        """DID must recover ATT ≈ 3.0 from known DGP."""
        df = _did_data()
        r = sp.did(df, y='y', treat='treat', time='post')
        assert r.estimate == pytest.approx(3.0, abs=0.2)

    def test_did_significance(self):
        """DID ATT should be significant at 1%."""
        df = _did_data()
        r = sp.did(df, y='y', treat='treat', time='post')
        assert r.pvalue < 0.01

    def test_did_ci_covers_true(self):
        """95% CI should contain the true ATT = 3.0."""
        df = _did_data()
        r = sp.did(df, y='y', treat='treat', time='post')
        assert r.ci[0] < 3.0 < r.ci[1]


# =====================================================================
# 5. RD — jump recovery
# =====================================================================

class TestRDValidation:
    """Validate sp.rdrobust() recovers known discontinuity."""

    def test_rd_estimate_recovery(self):
        """RD must recover jump ≈ 2.0 from known DGP."""
        df = _rd_data()
        r = sp.rdrobust(df, y='y', x='x', c=0)
        assert r.estimate == pytest.approx(2.0, abs=0.3)

    def test_rd_ci_covers_true(self):
        """95% CI should contain the true jump = 2.0."""
        df = _rd_data()
        r = sp.rdrobust(df, y='y', x='x', c=0)
        assert r.ci[0] < 2.0 < r.ci[1]

    def test_rd_bandwidth_positive(self):
        """Selected bandwidth must be positive."""
        df = _rd_data()
        r = sp.rdrobust(df, y='y', x='x', c=0)
        bw = r.model_info.get('bandwidth_h', r.model_info.get('bandwidth'))
        assert bw is not None and bw > 0


# =====================================================================
# 6. Matching — ATT recovery
# =====================================================================

class TestMatchValidation:
    """Validate sp.match() recovers known ATT."""

    def test_match_att_recovery(self):
        """Matching ATT must be close to true ATT = 2.0."""
        df = _match_data()
        r = sp.match(df, y='y', treat='treat', covariates=['x'])
        assert r.estimate == pytest.approx(2.0, abs=0.3)

    def test_match_significance(self):
        """Matched ATT should be significant at 5%."""
        df = _match_data()
        r = sp.match(df, y='y', treat='treat', covariates=['x'])
        assert r.pvalue < 0.05


# =====================================================================
# 7. DML — ATE recovery
# =====================================================================

class TestDMLValidation:
    """Validate sp.dml() recovers known ATE."""

    def test_dml_ate_recovery(self):
        """DML must recover ATE ≈ 2.0 from known DGP."""
        np.random.seed(5)
        n = 2000
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        treat = (np.random.randn(n) + 0.3 * x1 > 0).astype(int)
        y = 2.0 * treat + 1.0 * x1 + 0.5 * x2 + np.random.randn(n) * 0.5
        df = pd.DataFrame({'y': y, 'treat': treat, 'x1': x1, 'x2': x2})
        r = sp.dml(df, y='y', treat='treat', covariates=['x1', 'x2'])
        assert r.estimate == pytest.approx(2.0, abs=0.3)

    def test_dml_ci_covers_true(self):
        """95% CI should contain the true ATE = 2.0."""
        np.random.seed(5)
        n = 2000
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        treat = (np.random.randn(n) + 0.3 * x1 > 0).astype(int)
        y = 2.0 * treat + 1.0 * x1 + 0.5 * x2 + np.random.randn(n) * 0.5
        df = pd.DataFrame({'y': y, 'treat': treat, 'x1': x1, 'x2': x2})
        r = sp.dml(df, y='y', treat='treat', covariates=['x1', 'x2'])
        assert r.ci[0] < 2.0 < r.ci[1]


# =====================================================================
# 8. Cross-estimator consistency
# =====================================================================

class TestCrossEstimatorConsistency:
    """Different methods on the same DGP should agree directionally."""

    def test_ols_vs_panel_pooled(self):
        """Panel pooled OLS ≈ plain OLS on same data."""
        df = _panel_data()
        r_ols = sp.regress('y ~ x', data=df)
        r_pool = sp.panel(df, 'y ~ x', entity='id', time='time',
                          method='pooled')
        assert r_ols.params['x'] == pytest.approx(
            r_pool.params['x'], rel=0.01)

    def test_did_vs_regression_did(self):
        """DID estimate ≈ interaction coefficient from OLS."""
        df = _did_data()
        r_did = sp.did(df, y='y', treat='treat', time='post')
        r_ols = sp.regress('y ~ treat * post', data=df)
        # The interaction coefficient is the DID estimate
        interaction_key = [k for k in r_ols.params.index
                          if 'treat' in k and 'post' in k][0]
        assert r_did.estimate == pytest.approx(
            r_ols.params[interaction_key], abs=0.3)
