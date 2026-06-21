"""
Tests for CJM rddensity and read_data I/O.
"""

import pytest
import numpy as np
import pandas as pd
import subprocess

from statspai.diagnostics.rddensity import rddensity, _find_rscript
from statspai.utils.io import read_data
from statspai.core.results import CausalResult
from statspai.exceptions import DataInsufficient, MethodIncompatibility


@pytest.fixture
def clean_rd():
    """Uniform density — no manipulation."""
    rng = np.random.default_rng(42)
    X = rng.uniform(-2, 2, 5000)
    return pd.DataFrame({"x": X})


@pytest.fixture
def manipulated_rd():
    """Bunching above cutoff — manipulation."""
    rng = np.random.default_rng(42)
    X = rng.uniform(-2, 2, 5000)
    X_extra = rng.uniform(0, 0.3, 1500)
    return pd.DataFrame({"x": np.concatenate([X, X_extra])})


class TestRDDensity:
    def test_basic_run(self, clean_rd):
        result = rddensity(clean_rd, x="x", c=0)
        assert isinstance(result, CausalResult)
        assert "CJM" in result.method

    def test_clean_symmetric(self, clean_rd):
        """Uniform data: density estimates should be similar on both sides."""
        result = rddensity(clean_rd, x="x", c=0)
        f_l = result.model_info["density_left"]
        f_r = result.model_info["density_right"]
        # Densities should be within 50% of each other for uniform data
        ratio = max(f_l, f_r) / max(min(f_l, f_r), 1e-10)
        assert ratio < 2.0

    def test_manipulated_rejects(self, manipulated_rd):
        """Bunched data should reject H0."""
        result = rddensity(manipulated_rd, x="x", c=0)
        assert result.pvalue < 0.1

    def test_native_registered_examples_match_reference_defaults(self):
        """Native path ports the rddensity default selector/test."""
        import statspai as sp

        lee = sp.datasets.lee_2008_senate()
        lee_result = rddensity(lee, x="margin", c=0)
        assert lee_result.model_info["backend"] == "native"
        assert lee_result.model_info["bandwidth_source"] == "rddensity_comb"
        assert lee_result.model_info["validation_tier"] == "T2_native_reference_parity"
        assert lee_result.model_info["reference_backend"] == "rddensity"
        assert abs(lee_result.model_info["density_left"] - 1.7010417652833731) < 1e-9
        assert abs(lee_result.model_info["density_right"] - 1.7445202120700525) < 1e-9
        assert abs(lee_result.model_info["density_diff"] - 0.043478446786679337) < 1e-9
        assert abs(lee_result.model_info["bandwidth_left"] - 0.12670043574048026) < 1e-9
        assert (
            abs(lee_result.model_info["bandwidth_right"] - 0.11954302417477287) < 1e-9
        )
        assert abs(lee_result.pvalue - 0.85710438024631219) < 1e-9
        assert lee_result.pvalue > 0.1

        rng = np.random.default_rng(12345)
        base = rng.uniform(-2, 2, 5000)
        extra = rng.uniform(0, 0.25, 1750)
        bunched = pd.DataFrame({"x": np.concatenate([base, extra])})
        bunched_result = rddensity(bunched, x="x", c=0)
        assert bunched_result.pvalue < 0.001
        assert bunched_result.model_info["density_diff"] > 0.5

    def test_native_deterministic_grid_matches_reference_manual_bandwidth(self):
        """Manual bandwidth path matches rddensity on analytic-density grids."""
        symmetric = pd.DataFrame(
            {
                "x": np.concatenate(
                    [
                        np.linspace(-2.0, -1e-4, 4000),
                        np.linspace(0.0, 2.0, 4000),
                    ]
                )
            }
        )
        result = rddensity(symmetric, x="x", c=0.0, h=0.5)
        assert result.model_info["bandwidth_source"] == "manual_scalar"
        assert result.model_info["validation_tier"] == ("T2_native_reference_parity")
        assert abs(result.model_info["density_left"] - 0.2499812451550634) < 1e-9
        assert abs(result.model_info["density_right"] - 0.24996874609322706) < 1e-9
        assert abs(result.model_info["density_diff"] + 1.2499061836340752e-05) < 1e-9
        assert result.pvalue > 0.99

        bunched = pd.DataFrame(
            {
                "x": np.concatenate(
                    [
                        np.linspace(-2.0, -1e-4, 4000),
                        np.linspace(0.0, 2.0, 6000),
                    ]
                )
            }
        )
        bunched_result = rddensity(bunched, x="x", c=0.0, h=0.5)
        assert (
            abs(bunched_result.model_info["density_left"] - 0.19997999599915844) < 1e-9
        )
        assert (
            abs(bunched_result.model_info["density_right"] - 0.29997999799932201) < 1e-9
        )
        assert (
            abs(bunched_result.model_info["density_diff"] - 0.10000000200016357) < 1e-9
        )
        assert bunched_result.pvalue < 0.01

    def test_density_estimates(self, clean_rd):
        result = rddensity(clean_rd, x="x", c=0)
        assert result.model_info["density_left"] > 0
        assert result.model_info["density_right"] > 0

    def test_custom_bandwidth(self, clean_rd):
        result = rddensity(clean_rd, x="x", c=0, h=0.5)
        assert abs(result.model_info["bandwidth_left"] - 0.5) < 0.01
        assert result.model_info["bandwidth_source"] == "manual_scalar"

    def test_side_specific_bandwidth(self, clean_rd):
        result = rddensity(clean_rd, x="x", c=0, h=(0.35, 0.55))
        assert abs(result.model_info["bandwidth_left"] - 0.35) < 1e-12
        assert abs(result.model_info["bandwidth_right"] - 0.55) < 1e-12
        assert result.model_info["bandwidth_source"] == "manual_side_specific"

    def test_manual_bandwidth_keeps_native_reference_scope(self, clean_rd):
        result = rddensity(clean_rd, x="x", c=0, h=(0.35, 0.55))
        assert result.model_info["backend"] == "native"
        assert (
            "rdbwdensity combination bandwidths" in result.model_info["validation_note"]
        )
        assert "backend='r'" in result.model_info["validation_note"]

    def test_invalid_bandwidth(self, clean_rd):
        with pytest.raises(MethodIncompatibility, match="length-2"):
            rddensity(clean_rd, x="x", c=0, h=(0.2, 0.3, 0.4))
        with pytest.raises(MethodIncompatibility, match="positive"):
            rddensity(clean_rd, x="x", c=0, h=-0.2)

    def test_invalid_backend(self, clean_rd):
        with pytest.raises(MethodIncompatibility, match="backend"):
            rddensity(clean_rd, x="x", c=0, backend="unknown")

    def test_input_validation_taxonomy(self, clean_rd):
        with pytest.raises(MethodIncompatibility, match="Column"):
            rddensity(clean_rd, x="missing", c=0)

        with pytest.raises(MethodIncompatibility, match="p must"):
            rddensity(clean_rd, x="x", c=0, p=0)

        with pytest.raises(DataInsufficient, match="Need at least 20"):
            rddensity(clean_rd.head(10), x="x", c=0)

        one_sided = pd.DataFrame({"x": np.linspace(0.1, 2.0, 100)})
        with pytest.raises(DataInsufficient, match="each side"):
            rddensity(one_sided, x="x", c=0)

    def test_r_backend_matches_reference_package(self, clean_rd):
        rscript = _find_rscript()
        if rscript is None:
            pytest.skip("Rscript is not installed")
        probe = subprocess.run(
            [
                rscript,
                "-e",
                "quit(status = as.integer(!requireNamespace('rddensity', quietly=TRUE) || !requireNamespace('jsonlite', quietly=TRUE)))",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            pytest.skip("R packages rddensity/jsonlite are not installed")

        result = rddensity(clean_rd, x="x", c=0, backend="r")
        assert result.model_info["backend"] == "rddensity"
        assert result.model_info["bandwidth_source"] == "rddensity_default"
        assert result.model_info["validation_tier"] == "reference_backend_bridge"
        assert (
            "not counted as native Python parity evidence"
            in result.model_info["validation_note"]
        )
        assert np.isfinite(result.pvalue)

    def test_nonzero_cutoff(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 10, 3000)
        df = pd.DataFrame({"x": X})
        result = rddensity(df, x="x", c=5)
        assert isinstance(result, CausalResult)

    def test_cite(self, clean_rd):
        result = rddensity(clean_rd, x="x")
        assert "cattaneo" in result.cite().lower()


class TestReadData:
    def test_csv(self, tmp_path):
        """Read CSV file."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = str(tmp_path / "test.csv")
        df.to_csv(path, index=False)
        result = read_data(path)
        assert len(result) == 3
        assert "a" in result.columns

    def test_excel(self, tmp_path):
        """Read Excel file."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        path = str(tmp_path / "test.xlsx")
        df.to_excel(path, index=False)
        result = read_data(path)
        assert len(result) == 2

    def test_parquet(self, tmp_path):
        """Read Parquet file (requires pyarrow)."""
        pytest.importorskip("pyarrow")
        df = pd.DataFrame({"x": [1, 2, 3]})
        path = str(tmp_path / "test.parquet")
        df.to_parquet(path, index=False)
        result = read_data(path)
        assert len(result) == 3

    def test_unsupported_format(self, tmp_path):
        path = str(tmp_path / "test.xyz")
        with open(path, "w") as f:
            f.write("test")
        with pytest.raises(ValueError, match="Unsupported"):
            read_data(path)


class TestIntegration:
    def test_imports(self):
        import statspai as sp

        assert hasattr(sp, "rddensity")
        assert hasattr(sp, "read_data")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
