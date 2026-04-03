"""
Tests for statspai.fixest module — pyfixest integration.
"""

import pytest
import numpy as np
import pandas as pd

# Skip the entire module if pyfixest is not installed
pyfixest = pytest.importorskip("pyfixest")

from statspai.fixest import feols, fepois, feglm, etable
from statspai.core.results import EconometricResults
from statspai.output.outreg2 import outreg2


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def panel_data():
    """Generate a simple panel dataset."""
    np.random.seed(42)
    n_firms = 50
    n_years = 10
    n = n_firms * n_years

    firm_id = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(2010, 2010 + n_years), n_firms)
    firm_effect = np.repeat(np.random.randn(n_firms) * 2, n_years)
    year_effect = np.tile(np.random.randn(n_years), n_firms)

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + firm_effect + year_effect + np.random.randn(n)

    return pd.DataFrame({
        "y": y,
        "x1": x1,
        "x2": x2,
        "firm_id": firm_id,
        "year": year,
    })


@pytest.fixture
def count_data():
    """Generate count data for Poisson regression."""
    np.random.seed(123)
    n = 500
    group = np.random.randint(0, 10, n)
    x1 = np.random.randn(n)
    lam = np.exp(0.5 + 0.3 * x1 + 0.1 * group)
    y = np.random.poisson(lam)

    return pd.DataFrame({
        "y": y,
        "x1": x1,
        "group_id": group,
    })


# --------------------------------------------------------------------------- #
#  feols tests
# --------------------------------------------------------------------------- #

class TestFeols:
    """Tests for feols wrapper."""

    def test_simple_ols(self, panel_data):
        """Plain OLS without fixed effects."""
        result = feols("y ~ x1 + x2", data=panel_data)
        assert isinstance(result, EconometricResults)
        assert "x1" in result.params.index
        assert "x2" in result.params.index
        assert result.params["x1"] > 0  # true coef is 2.0

    def test_one_way_fe(self, panel_data):
        """OLS with one-way fixed effects."""
        result = feols("y ~ x1 + x2 | firm_id", data=panel_data)
        assert isinstance(result, EconometricResults)
        assert "x1" in result.params.index
        # FE absorbed, should not appear in params
        assert "firm_id" not in result.params.index
        assert abs(result.params["x1"] - 2.0) < 0.5

    def test_two_way_fe(self, panel_data):
        """OLS with two-way fixed effects."""
        result = feols("y ~ x1 + x2 | firm_id + year", data=panel_data)
        assert isinstance(result, EconometricResults)
        assert abs(result.params["x1"] - 2.0) < 0.5
        assert abs(result.params["x2"] - (-0.5)) < 0.5

    def test_clustered_se(self, panel_data):
        """Clustered standard errors."""
        result = feols(
            "y ~ x1 + x2 | firm_id",
            data=panel_data,
            vcov={"CRV1": "firm_id"},
        )
        assert isinstance(result, EconometricResults)
        assert all(result.std_errors > 0)

    def test_heteroskedastic_robust(self, panel_data):
        """HC1 robust standard errors."""
        result = feols("y ~ x1 + x2", data=panel_data, vcov="HC1")
        assert isinstance(result, EconometricResults)
        assert all(result.std_errors > 0)

    def test_result_has_diagnostics(self, panel_data):
        """Result should contain R-squared and RMSE."""
        result = feols("y ~ x1 + x2 | firm_id", data=panel_data)
        assert "R-squared" in result.diagnostics or "R-squared (within)" in result.diagnostics

    def test_result_has_nobs(self, panel_data):
        """Data info should contain nobs."""
        result = feols("y ~ x1 + x2", data=panel_data)
        assert result.data_info["nobs"] == len(panel_data)

    def test_summary_runs(self, panel_data):
        """summary() should produce a non-empty string."""
        result = feols("y ~ x1 + x2 | firm_id", data=panel_data)
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 0
        assert "x1" in s

    def test_pyfixest_fit_attached(self, panel_data):
        """Original pyfixest fit should be accessible."""
        result = feols("y ~ x1 + x2 | firm_id", data=panel_data)
        assert hasattr(result, "_pyfixest_fit")
        assert result._pyfixest_fit is not None

    def test_multiple_estimation(self, panel_data):
        """csw0 syntax should return list of EconometricResults."""
        results = feols("y ~ x1 | csw0(firm_id, year)", data=panel_data)
        assert isinstance(results, list)
        assert len(results) >= 2
        assert all(isinstance(r, EconometricResults) for r in results)


# --------------------------------------------------------------------------- #
#  fepois tests
# --------------------------------------------------------------------------- #

class TestFepois:
    """Tests for fepois wrapper."""

    def test_simple_poisson(self, count_data):
        """Poisson regression without fixed effects."""
        result = fepois("y ~ x1", data=count_data)
        assert isinstance(result, EconometricResults)
        assert "x1" in result.params.index

    def test_poisson_with_fe(self, count_data):
        """Poisson regression with fixed effects."""
        result = fepois("y ~ x1 | group_id", data=count_data)
        assert isinstance(result, EconometricResults)
        assert "x1" in result.params.index
        assert result.params["x1"] > 0  # true coef is 0.3


# --------------------------------------------------------------------------- #
#  outreg2 integration tests
# --------------------------------------------------------------------------- #

class TestOutreg2Integration:
    """Test that pyfixest results integrate with outreg2."""

    def test_outreg2_single_model(self, panel_data, tmp_path):
        """Export a single pyfixest result to Excel."""
        result = feols("y ~ x1 + x2 | firm_id", data=panel_data)
        outfile = str(tmp_path / "test_fixest.xlsx")
        outreg2(result, filename=outfile)
        import os
        assert os.path.exists(outfile)

    def test_outreg2_multiple_models(self, panel_data, tmp_path):
        """Export multiple pyfixest results to one table."""
        r1 = feols("y ~ x1", data=panel_data)
        r2 = feols("y ~ x1 + x2", data=panel_data)
        r3 = feols("y ~ x1 + x2 | firm_id", data=panel_data)
        outfile = str(tmp_path / "test_multi.xlsx")
        outreg2(r1, r2, r3, filename=outfile,
                model_names=["(1)", "(2)", "(3)"])
        import os
        assert os.path.exists(outfile)


# --------------------------------------------------------------------------- #
#  etable tests
# --------------------------------------------------------------------------- #

class TestEtable:
    """Tests for the etable convenience wrapper."""

    def test_etable_pyfixest_native(self, panel_data):
        """etable should use pyfixest's native etable when possible."""
        r1 = feols("y ~ x1", data=panel_data)
        r2 = feols("y ~ x1 + x2 | firm_id", data=panel_data)
        table = etable(r1, r2)
        assert table is not None
