"""Round-5 estimator provenance instrumentation.

Layered on Phases 3+4+7+8 (36 estimators). This round adds 12 ML-causal
+ identification-strategy estimators. Coverage 36/925 → **48/925**.

ML-causal:
- ``sp.tmle`` — van der Laan & Rose Targeted MLE.
- ``sp.tmle.ltmle`` — Longitudinal TMLE.
- ``sp.tmle.hal_tmle`` — TMLE with HAL nuisance.
- ``sp.causal_forest`` — GRF causal forest.
- ``sp.multi_arm_forest`` — multi-arm causal forest (Athey et al.).
- ``sp.iv_forest`` — Athey-Tibshirani-Wager IV forest.
- ``sp.metalearner`` — S/T/X/R/DR meta-learner dispatcher.
- ``sp.bcf`` — Hahn-Murray-Carvalho Bayesian Causal Forest.

Classical identification:
- ``sp.aipw`` — Augmented IPW (doubly robust, cross-fit).
- ``sp.ipw`` — Inverse Probability Weighting.
- ``sp.g_computation`` — parametric g-formula.
- ``sp.front_door`` — Pearl front-door adjustment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def binary_df():
    rng = np.random.default_rng(0)
    n = 250
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    ps = 1 / (1 + np.exp(-(X1 + X2)))
    d = (rng.uniform(size=n) < ps).astype(int)
    y = 0.5 * d + 0.3 * X1 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": X1, "x2": X2})


@pytest.fixture
def iv_df():
    rng = np.random.default_rng(2)
    n = 250
    z = rng.normal(size=n)
    d = rng.binomial(1, 0.5, size=n)
    y = 0.5 * d + 0.3 * z + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "z": z,
                          "x1": rng.normal(size=n),
                          "x2": rng.normal(size=n)})


@pytest.fixture
def front_door_df():
    rng = np.random.default_rng(3)
    n = 200
    d = rng.binomial(1, 0.5, size=n)
    m = 0.5 * d + rng.normal(size=n)
    y = 0.7 * m + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "m": m})


# ---------------------------------------------------------------------------
# ML-causal
# ---------------------------------------------------------------------------

class TestTmleProvenance:
    def test_attached(self, binary_df):
        r = sp.tmle(binary_df, y="y", treat="d",
                    covariates=["x1", "x2"], n_folds=3)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.tmle"
        assert prov.params["n_folds"] == 3
        assert prov.params["estimand"] == "ATE"


class TestHalTmleProvenance:
    def test_attached(self, binary_df):
        from statspai.tmle.hal_tmle import hal_tmle
        r = hal_tmle(binary_df, y="y", treat="d",
                      covariates=["x1", "x2"],
                      max_anchors_per_col=5, n_folds=3)
        prov = sp.get_provenance(r)
        assert prov is not None
        # hal_tmle delegates to tmle internally; the inner sp.tmle
        # provenance wins (overwrite=False) — both function names are
        # acceptable.
        assert prov.function in {"sp.tmle.hal_tmle", "sp.tmle"}


class TestLtmleProvenance:
    def test_attached(self):
        rng = np.random.default_rng(7)
        n = 200
        df = pd.DataFrame({
            "A0": rng.binomial(1, 0.5, size=n),
            "A1": rng.binomial(1, 0.5, size=n),
            "Y": rng.normal(size=n),
            "L0": rng.normal(size=n),
            "L1": rng.normal(size=n),
        })
        from statspai.tmle.ltmle import ltmle
        r = ltmle(df, y="Y", treatments=["A0", "A1"],
                   covariates_time=[["L0"], ["L1"]],
                   regime_treated=(1, 1), regime_control=(0, 0))
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.tmle.ltmle"
        assert prov.params["treatments"] == ["A0", "A1"]


class TestCausalForestProvenance:
    def test_attached(self, binary_df):
        cf = sp.causal_forest("y ~ d | x1 + x2", data=binary_df,
                                n_estimators=10)
        prov = sp.get_provenance(cf)
        assert prov is not None
        assert prov.function == "sp.causal_forest"
        assert prov.params["n_estimators"] == 10


class TestMultiArmForestProvenance:
    def test_attached(self, binary_df):
        rng = np.random.default_rng(4)
        df = binary_df.copy()
        df["d"] = rng.integers(0, 3, size=len(df))
        r = sp.multi_arm_forest(df, y="y", treat="d",
                                  covariates=["x1", "x2"],
                                  n_trees=20)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.multi_arm_forest"


class TestIvForestProvenance:
    def test_attached(self, iv_df):
        r = sp.iv_forest(iv_df, y="y", treat="d", instrument="z",
                          covariates=["x1", "x2"],
                          n_trees=20, n_bootstrap=10)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.iv_forest"


class TestMetalearnerProvenance:
    def test_attached(self, binary_df):
        from statspai.metalearners.metalearners import metalearner
        r = metalearner(binary_df, y="y", treat="d",
                          covariates=["x1", "x2"],
                          n_bootstrap=20, n_folds=2)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.metalearner"
        assert prov.params["learner"] == "dr"

    def test_learner_choice_captured(self, binary_df):
        from statspai.metalearners.metalearners import metalearner
        r = metalearner(binary_df, y="y", treat="d",
                          covariates=["x1", "x2"],
                          learner="t",
                          n_bootstrap=10, n_folds=2)
        prov = sp.get_provenance(r)
        assert prov.params["learner"] == "t"


class TestBcfProvenance:
    def test_attached(self, binary_df):
        r = sp.bcf(binary_df, y="y", treat="d",
                    covariates=["x1", "x2"],
                    n_bootstrap=20, n_folds=2)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.bcf"


# ---------------------------------------------------------------------------
# Classical identification
# ---------------------------------------------------------------------------

class TestAipwProvenance:
    def test_attached(self, binary_df):
        r = sp.aipw(binary_df, y="y", treat="d",
                     covariates=["x1", "x2"], n_folds=3)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.aipw"
        assert prov.params["estimand"] == "ATE"


class TestIpwProvenance:
    def test_attached(self, binary_df):
        r = sp.ipw(binary_df, y="y", treat="d",
                    covariates=["x1", "x2"],
                    n_bootstrap=20)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.ipw"


class TestGComputationProvenance:
    def test_attached(self, binary_df):
        r = sp.g_computation(binary_df, y="y", treat="d",
                              covariates=["x1", "x2"],
                              n_boot=20)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.g_computation"


class TestFrontDoorProvenance:
    def test_attached(self, front_door_df):
        r = sp.front_door(front_door_df, y="y", treat="d",
                            mediator="m", n_boot=20)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.front_door"
        assert prov.params["mediator"] == "m"


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestRound5LineageIntegration:
    def test_multi_estimator_pack(self, binary_df, tmp_path):
        # Three different ML-causal / classical estimators packed.
        r1 = sp.tmle(binary_df, y="y", treat="d",
                      covariates=["x1", "x2"], n_folds=2)
        r2 = sp.aipw(binary_df, y="y", treat="d",
                      covariates=["x1", "x2"], n_folds=2)
        r3 = sp.ipw(binary_df, y="y", treat="d",
                     covariates=["x1", "x2"],
                     n_bootstrap=20)

        rp = sp.replication_pack(
            [r1, r2, r3], tmp_path / "round5.zip",
            data=binary_df, env=False,
        )
        import json
        import zipfile
        with zipfile.ZipFile(rp.output_path) as zf:
            assert "lineage.json" in zf.namelist()
            lin = json.loads(zf.read("lineage.json"))
            assert lin["n_runs"] >= 3
            funcs = {v["function"] for v in lin["runs"].values()}
            assert "sp.tmle" in funcs
            assert "sp.aipw" in funcs
            assert "sp.ipw" in funcs
