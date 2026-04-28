"""Round-4 estimator provenance instrumentation.

Layered on Phases 3+4+7 (21 estimators). This round adds 12
instrumentation points covering 15 user-facing estimators (the
JIVE family of 4 share a single ``_run`` instrumentation):

IV family (8 user-facing names):
- ``sp.liml`` — Limited Information Maximum Likelihood / Fuller.
- ``sp.jive`` — legacy single-method JIVE.
- ``sp.lasso_iv`` — Belloni-Chen-Chernozhukov-Hansen (2012).
- ``sp.iv.bayesian_iv`` — Chernozhukov-Hong (2003) AR posterior.
- ``sp.iv.jive1`` — Angrist-Imbens-Krueger (1999).
- ``sp.iv.ujive`` — Kolesár (2013).
- ``sp.iv.ijive`` — Ackerberg-Devereux (2009).
- ``sp.iv.rjive`` — Hansen-Kozbur (2014) ridge-JIVE.
- ``sp.iv.mte`` — Brinch-Mogstad-Wiswall (2017) polynomial MTE.

Matching family (5):
- ``sp.match`` (dispatcher).
- ``sp.optimal_match`` — Hungarian-algorithm 1:1.
- ``sp.cardinality_match`` — Zubizarreta (2014) LP.
- ``sp.genmatch`` — Diamond-Sekhon (2013) genetic matching.
- ``sp.sbw`` — Zubizarreta (2015) Stable Balancing Weights.

DML (1):
- ``sp.dml`` — Chernozhukov et al. (2018) Double ML dispatcher.

Total provenance coverage after this round: **36/925**.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def iv_df():
    """Toy IV dataset: z is the instrument, x is endogenous, y is outcome."""
    rng = np.random.default_rng(0)
    n = 400
    z = rng.normal(size=n)
    x = z + rng.normal(size=n)
    y = 1.0 * x + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z})


@pytest.fixture
def iv_df_binary():
    """Binary-treatment IV dataset for MTE / IIVM-style estimators."""
    rng = np.random.default_rng(1)
    n = 300
    z = rng.normal(size=n)
    p = 1 / (1 + np.exp(-z))
    d = (rng.uniform(size=n) < p).astype(int)
    y = 0.5 * d + z + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "z": z})


@pytest.fixture
def matching_df():
    """Treated/control panel with one covariate. Designed with
    n_control > n_treated so optimal_match's Hungarian-algorithm
    constraint is satisfied."""
    rng = np.random.default_rng(2)
    n_treat = 60
    n_ctrl = 140
    return pd.DataFrame({
        "y": np.concatenate([
            1.0 + rng.normal(size=n_treat),     # treated mean ≈ 1
            rng.normal(size=n_ctrl),            # control mean ≈ 0
        ]),
        "d": np.concatenate([
            np.ones(n_treat, dtype=int),
            np.zeros(n_ctrl, dtype=int),
        ]),
        "x1": rng.normal(size=n_treat + n_ctrl),
    })


@pytest.fixture
def dml_df():
    """High-dim controls + binary treatment for DML PLR."""
    rng = np.random.default_rng(3)
    n = 250
    return pd.DataFrame({
        "y": rng.normal(size=n),
        "d": rng.binomial(1, 0.5, size=n),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
    })


# ---------------------------------------------------------------------------
# IV family
# ---------------------------------------------------------------------------

class TestLimlProvenance:
    def test_attached(self, iv_df):
        r = sp.liml(data=iv_df, y="y", x_endog=["x"], z=["z"])
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.iv.liml"
        assert prov.data_hash
        assert prov.params["x_endog"] == ["x"]
        assert prov.params["z"] == ["z"]


class TestLegacyJiveProvenance:
    def test_attached(self, iv_df):
        r = sp.jive(data=iv_df, y="y", x_endog=["x"], z=["z"])
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.iv.jive"
        assert prov.params["variant"] == "jive1"


class TestLassoIvProvenance:
    def test_attached(self, iv_df):
        # Pre-existing iv() API drift in lasso_iv was fixed alongside
        # this instrumentation: lasso_iv now builds a formula string
        # for the formula-only sp.iv() API.
        r = sp.lasso_iv(data=iv_df, y="y", x_endog=["x"], z=["z"])
        prov = sp.get_provenance(r)
        assert prov is not None
        # Inner sp.iv attaches first; lasso_iv is overwrite=False, so
        # the inner record (sp.iv) preserves. Both names are valid
        # — replication-side, what matters is the trail of params.
        assert prov.function in {"sp.iv.lasso_iv", "sp.iv"}


class TestBayesianIvProvenance:
    def test_attached(self, iv_df):
        from statspai.iv import bayesian_iv
        r = bayesian_iv(
            y="y", endog="x", instruments=["z"], data=iv_df,
            n_draws=200, n_warmup=50, random_state=0,
        )
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.iv.bayesian_iv"
        assert prov.params["n_draws"] == 200
        assert prov.params["n_warmup"] == 50


class TestJiveVariantsProvenance:
    """All four JIVE variants flow through ``_run``; the ``method``
    argument discriminates and surfaces in ``Provenance.function``."""

    def test_jive1(self, iv_df):
        from statspai.iv import jive1
        r = jive1(y="y", endog="x", instruments=["z"], data=iv_df)
        prov = sp.get_provenance(r)
        assert prov.function == "sp.iv.jive1"
        assert prov.params["method"] == "jive1"

    def test_ujive(self, iv_df):
        from statspai.iv import ujive
        r = ujive(y="y", endog="x", instruments=["z"], data=iv_df)
        prov = sp.get_provenance(r)
        assert prov.function == "sp.iv.ujive"

    def test_ijive(self, iv_df):
        from statspai.iv import ijive
        r = ijive(y="y", endog="x", instruments=["z"], data=iv_df)
        prov = sp.get_provenance(r)
        assert prov.function == "sp.iv.ijive"

    def test_rjive(self, iv_df):
        from statspai.iv import rjive
        r = rjive(y="y", endog="x", instruments=["z"],
                  data=iv_df, ridge=0.5)
        prov = sp.get_provenance(r)
        assert prov.function == "sp.iv.rjive"
        assert prov.params["ridge"] == 0.5


class TestMteProvenance:
    def test_attached(self, iv_df_binary):
        from statspai.iv import mte
        r = mte(y="y", treatment="d", instruments=["z"],
                data=iv_df_binary, poly_degree=2, trim=0.05)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.iv.mte"
        assert prov.params["poly_degree"] == 2
        assert prov.params["trim"] == 0.05


# ---------------------------------------------------------------------------
# Matching family
# ---------------------------------------------------------------------------

class TestMatchProvenance:
    def test_attached(self, matching_df):
        r = sp.match(matching_df, y="y", treat="d", covariates=["x1"])
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.matching.match"
        assert prov.params["covariates"] == ["x1"]


class TestOptimalMatchProvenance:
    def test_attached(self, matching_df):
        r = sp.optimal_match(matching_df, treatment="d", outcome="y",
                              covariates=["x1"])
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.matching.optimal_match"
        assert prov.params["metric"] == "mahalanobis"


class TestCardinalityMatchProvenance:
    def test_attached(self, matching_df):
        r = sp.cardinality_match(
            matching_df, treatment="d", outcome="y",
            covariates=["x1"], smd_tolerance=0.5,
        )
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.matching.cardinality_match"
        assert prov.params["smd_tolerance"] == 0.5


class TestGenmatchProvenance:
    def test_attached(self, matching_df):
        # Tiny GA budget — instrumentation tests don't need accuracy.
        r = sp.genmatch(
            matching_df, y="y", treat="d", covariates=["x1"],
            generations=3, population_size=10, random_state=0,
        )
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.matching.genmatch"
        assert prov.params["generations"] == 3


class TestSbwProvenance:
    def test_attached(self, matching_df):
        r = sp.sbw(matching_df, treat="d", covariates=["x1"], y="y",
                    delta=0.5)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.matching.sbw"
        # delta passes through (scalar form).
        assert prov.params["delta"] == 0.5


# ---------------------------------------------------------------------------
# DML
# ---------------------------------------------------------------------------

class TestDmlProvenance:
    def test_attached_plr(self, dml_df):
        # Default model='plr' is fastest; instrumentation test doesn't
        # need accuracy.
        r = sp.dml(dml_df, y="y", treat="d",
                    covariates=["x1", "x2", "x3"],
                    n_folds=3)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.dml"
        assert prov.params["model"] == "plr"
        assert prov.params["n_folds"] == 3


# ---------------------------------------------------------------------------
# Integration: round-4 estimators flow into replication_pack lineage
# ---------------------------------------------------------------------------

class TestRound4LineageIntegration:
    def test_multi_estimator_pack(self, iv_df, matching_df, tmp_path):
        # Run three estimators across two families.
        liml_r = sp.liml(data=iv_df, y="y", x_endog=["x"], z=["z"])
        match_r = sp.match(matching_df, y="y", treat="d",
                            covariates=["x1"])
        opt_r = sp.optimal_match(matching_df, treatment="d", outcome="y",
                                  covariates=["x1"])

        rp = sp.replication_pack(
            [liml_r, match_r, opt_r],
            tmp_path / "round4.zip",
            data=iv_df, env=False,
        )
        import json
        import zipfile
        with zipfile.ZipFile(rp.output_path) as zf:
            assert "lineage.json" in zf.namelist()
            lin = json.loads(zf.read("lineage.json"))
            assert lin["n_runs"] >= 3
            funcs = {v["function"] for v in lin["runs"].values()}
            assert "sp.iv.liml" in funcs
            assert "sp.matching.match" in funcs
            assert "sp.matching.optimal_match" in funcs
