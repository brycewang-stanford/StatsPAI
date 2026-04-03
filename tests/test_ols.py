"""
Tests for OLS regression functionality
"""

import pytest
import numpy as np
import pandas as pd
from statspai import regress
from statspai.regression.ols import OLSRegression


class TestOLSRegression:
    """Test cases for OLS regression"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n = 100
        
        # Generate data
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        epsilon = np.random.normal(0, 1, n)
        y = 1 + 2*x1 + 3*x2 + epsilon
        
        df = pd.DataFrame({
            'y': y,
            'x1': x1,
            'x2': x2,
            'group': np.random.choice(['A', 'B', 'C'], n)
        })
        
        return df
    
    def test_basic_ols(self, sample_data):
        """Test basic OLS functionality"""
        results = regress("y ~ x1 + x2", data=sample_data)
        
        # Check that results object is created
        assert results is not None
        assert len(results.params) == 3  # Intercept + 2 variables
        
        # Check parameter names
        expected_names = ['Intercept', 'x1', 'x2']
        assert list(results.params.index) == expected_names
        
        # Check that coefficients are reasonable (given true values 1, 2, 3)
        assert abs(results.params['Intercept'] - 1) < 0.5
        assert abs(results.params['x1'] - 2) < 0.5
        assert abs(results.params['x2'] - 3) < 0.5
    
    def test_robust_standard_errors(self, sample_data):
        """Test robust standard errors"""
        results_nonrobust = regress("y ~ x1 + x2", data=sample_data, robust='nonrobust')
        results_hc1 = regress("y ~ x1 + x2", data=sample_data, robust='hc1')
        
        # Standard errors should be different
        assert not np.allclose(results_nonrobust.std_errors, results_hc1.std_errors)
        
        # But parameters should be the same
        assert np.allclose(results_nonrobust.params, results_hc1.params)
    
    def test_cluster_standard_errors(self, sample_data):
        """Test clustered standard errors"""
        results = regress("y ~ x1 + x2", data=sample_data, cluster='group')
        
        # Should run without error
        assert results is not None
        assert len(results.params) == 3
    
    def test_summary_output(self, sample_data):
        """Test summary output generation"""
        results = regress("y ~ x1 + x2", data=sample_data)
        summary = results.summary()
        
        # Check that summary is a string and contains expected elements
        assert isinstance(summary, str)
        assert 'OLS' in summary
        assert 'Coefficient' in summary
        assert 'Std. Error' in summary
        assert 'R-squared' in summary
    
    def test_model_class_interface(self, sample_data):
        """Test the model class interface"""
        model = OLSRegression(formula="y ~ x1 + x2", data=sample_data)
        results = model.fit()
        
        assert model.is_fitted
        assert results is not None
        
        # Test prediction on training data
        predictions = model.predict()
        assert len(predictions) == len(sample_data)
    
    def test_no_constant(self, sample_data):
        """Test regression without constant term"""
        results = regress("y ~ x1 + x2 - 1", data=sample_data)
        
        # Should have only 2 parameters (no intercept)
        assert len(results.params) == 2
        assert 'Intercept' not in results.params.index
    
    def test_confidence_intervals(self, sample_data):
        """Test confidence interval calculation"""
        results = regress("y ~ x1 + x2", data=sample_data)
        conf_int = results.conf_int()
        
        # Check structure
        assert isinstance(conf_int, pd.DataFrame)
        assert conf_int.shape[0] == len(results.params)
        assert conf_int.shape[1] == 2  # Lower and upper bounds
        
        # Check that confidence intervals make sense
        for param in results.params.index:
            lower = conf_int.loc[param].iloc[0]
            upper = conf_int.loc[param].iloc[1]
            estimate = results.params[param]
            
            assert lower < estimate < upper


if __name__ == "__main__":
    pytest.main([__file__])
