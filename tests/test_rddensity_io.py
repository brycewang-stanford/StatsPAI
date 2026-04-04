"""
Tests for CJM rddensity and read_data I/O.
"""

import pytest
import numpy as np
import pandas as pd

from statspai.diagnostics.rddensity import rddensity
from statspai.utils.io import read_data
from statspai.core.results import CausalResult


@pytest.fixture
def clean_rd():
    """Uniform density — no manipulation."""
    rng = np.random.default_rng(42)
    X = rng.uniform(-2, 2, 5000)
    return pd.DataFrame({'x': X})


@pytest.fixture
def manipulated_rd():
    """Bunching above cutoff — manipulation."""
    rng = np.random.default_rng(42)
    X = rng.uniform(-2, 2, 5000)
    X_extra = rng.uniform(0, 0.3, 1500)
    return pd.DataFrame({'x': np.concatenate([X, X_extra])})


class TestRDDensity:
    def test_basic_run(self, clean_rd):
        result = rddensity(clean_rd, x='x', c=0)
        assert isinstance(result, CausalResult)
        assert 'CJM' in result.method

    def test_clean_symmetric(self, clean_rd):
        """Uniform data: density estimates should be similar on both sides."""
        result = rddensity(clean_rd, x='x', c=0)
        f_l = result.model_info['density_left']
        f_r = result.model_info['density_right']
        # Densities should be within 50% of each other for uniform data
        ratio = max(f_l, f_r) / max(min(f_l, f_r), 1e-10)
        assert ratio < 2.0

    def test_manipulated_rejects(self, manipulated_rd):
        """Bunched data should reject H0."""
        result = rddensity(manipulated_rd, x='x', c=0)
        assert result.pvalue < 0.1

    def test_density_estimates(self, clean_rd):
        result = rddensity(clean_rd, x='x', c=0)
        assert result.model_info['density_left'] > 0
        assert result.model_info['density_right'] > 0

    def test_custom_bandwidth(self, clean_rd):
        result = rddensity(clean_rd, x='x', c=0, h=0.5)
        assert abs(result.model_info['bandwidth_left'] - 0.5) < 0.01

    def test_nonzero_cutoff(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 10, 3000)
        df = pd.DataFrame({'x': X})
        result = rddensity(df, x='x', c=5)
        assert isinstance(result, CausalResult)

    def test_cite(self, clean_rd):
        result = rddensity(clean_rd, x='x')
        assert 'cattaneo' in result.cite().lower()


class TestReadData:
    def test_csv(self, tmp_path):
        """Read CSV file."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        path = str(tmp_path / 'test.csv')
        df.to_csv(path, index=False)
        result = read_data(path)
        assert len(result) == 3
        assert 'a' in result.columns

    def test_excel(self, tmp_path):
        """Read Excel file."""
        df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
        path = str(tmp_path / 'test.xlsx')
        df.to_excel(path, index=False)
        result = read_data(path)
        assert len(result) == 2

    def test_parquet(self, tmp_path):
        """Read Parquet file (requires pyarrow)."""
        pytest.importorskip('pyarrow')
        df = pd.DataFrame({'x': [1, 2, 3]})
        path = str(tmp_path / 'test.parquet')
        df.to_parquet(path, index=False)
        result = read_data(path)
        assert len(result) == 3

    def test_unsupported_format(self, tmp_path):
        path = str(tmp_path / 'test.xyz')
        with open(path, 'w') as f:
            f.write('test')
        with pytest.raises(ValueError, match="Unsupported"):
            read_data(path)


class TestIntegration:
    def test_imports(self):
        import statspai as sp
        assert hasattr(sp, 'rddensity')
        assert hasattr(sp, 'read_data')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
