"""Round-8 estimator provenance instrumentation.

Layered on Phases 3+4+7+8+9+10+11 (61 estimators). This round adds 5
more spanning bounds / randomization inference / imputation. Coverage
61/925 → **66/925**.

Estimators (5):
- ``sp.balke_pearl`` — Balke-Pearl bounds on ATE.
- ``sp.lee_bounds`` — Lee (2009) trimming bounds under selection.
- ``sp.manski_bounds`` — Manski (1990) worst-case bounds.
- ``sp.fisher_exact`` — Fisher randomization test (permutation).
- ``sp.imputation.mice`` — Multiple Imputation by Chained Equations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def binary_df():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "y": rng.binomial(1, 0.5, size=n),
        "d": rng.binomial(1, 0.5, size=n),
        "z": rng.binomial(1, 0.5, size=n),
    })


class TestFisherExactProvenance:
    def test_attached(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"y": rng.normal(size=150),
                            "d": rng.binomial(1, 0.5, size=150)})
        r = sp.fisher_exact(df, y="y", treatment="d", n_perm=200)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.inference.fisher_exact"
        assert prov.params["n_perm"] == 200


class TestBalkePearlProvenance:
    def test_attached(self, binary_df):
        r = sp.balke_pearl(binary_df, y="y", treat="d",
                            instrument="z")
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.bounds.balke_pearl"


class TestLeeBoundsProvenance:
    def test_attached(self):
        rng = np.random.default_rng(2)
        n = 200
        df = pd.DataFrame({
            "y": rng.normal(size=n),
            "d": rng.binomial(1, 0.5, size=n),
            "s": rng.binomial(1, 0.7, size=n),
        })
        df.loc[df["s"] == 0, "y"] = np.nan
        r = sp.lee_bounds(df, y="y", treat="d", selection="s",
                           n_bootstrap=20)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.bounds.lee_bounds"


class TestManskiBoundsProvenance:
    def test_attached(self):
        rng = np.random.default_rng(3)
        n = 200
        df = pd.DataFrame({
            "y": rng.normal(size=n),
            "d": rng.binomial(1, 0.5, size=n),
        })
        r = sp.manski_bounds(df, y="y", treat="d",
                              y_lower=-3.0, y_upper=3.0,
                              n_bootstrap=20)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.bounds.manski_bounds"


class TestMiceProvenance:
    def test_attached(self):
        rng = np.random.default_rng(4)
        n = 200
        df = pd.DataFrame({
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
        })
        df.loc[rng.uniform(size=n) < 0.1, "a"] = np.nan
        from statspai.imputation.mice import mice
        r = mice(df, m=2, max_iter=2)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.imputation.mice"
        assert prov.params["m"] == 2


class TestRound8LineageIntegration:
    def test_multi_estimator_pack(self, binary_df, tmp_path):
        rng = np.random.default_rng(5)
        n = 200
        df_y = pd.DataFrame({"y": rng.normal(size=n),
                              "d": rng.binomial(1, 0.5, size=n)})
        r1 = sp.balke_pearl(binary_df, y="y", treat="d",
                              instrument="z")
        r2 = sp.fisher_exact(df_y, y="y", treatment="d", n_perm=100)

        rp = sp.replication_pack(
            [r1, r2], tmp_path / "round8.zip",
            data=binary_df, env=False,
        )
        import json
        import zipfile
        with zipfile.ZipFile(rp.output_path) as zf:
            assert "lineage.json" in zf.namelist()
            lin = json.loads(zf.read("lineage.json"))
            funcs = {v["function"] for v in lin["runs"].values()}
            assert "sp.bounds.balke_pearl" in funcs
            assert "sp.inference.fisher_exact" in funcs
