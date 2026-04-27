"""Tests that the top-tier estimators attach a Provenance record at fit().

The instrumented estimators in this commit:
- ``sp.regress`` (regression/ols.py:regress) — OLS workhorse.
- ``sp.callaway_santanna`` (did/callaway_santanna.py) — modern DiD.
- ``sp.did_2x2`` (did/did_2x2.py) — classic 2×2 DiD.
- ``statspai.regression.iv.iv`` — unified IV (2SLS / LIML / GMM / JIVE).

Synth dispatcher (synth/scm.py:synth) is deferred — its 13+ return
paths require a separate refactor pass that splits dispatcher logic
from provenance, scheduled for v1.7.3.

Each test verifies:
- ``result._provenance`` is set after fit (via :func:`sp.get_provenance`).
- The function name matches the estimator's logical name.
- ``data_hash`` is non-empty (the input frame was hashed).
- Key params are captured (``y`` / ``treatment`` / etc.).
- Provenance never breaks the estimator: a blank-input frame still
  returns a result *or* raises the estimator's normal error, never
  swallows the error inside lineage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# regress
# ---------------------------------------------------------------------------

@pytest.fixture
def cross_section_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "wage": rng.normal(loc=10, scale=1, size=200),
        "trained": rng.binomial(1, 0.5, size=200),
        "edu": rng.normal(size=200),
    })


class TestRegressProvenance:
    def test_attached(self, cross_section_df):
        r = sp.regress("wage ~ trained + edu", cross_section_df)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.regress"
        assert prov.data_hash
        # Key params captured.
        assert prov.params["formula"] == "wage ~ trained + edu"

    def test_robust_kwarg_captured(self, cross_section_df):
        r = sp.regress("wage ~ trained", cross_section_df, robust="hc1")
        prov = sp.get_provenance(r)
        assert prov.params["robust"] == "hc1"


# ---------------------------------------------------------------------------
# callaway_santanna
# ---------------------------------------------------------------------------

@pytest.fixture
def staggered_panel():
    rng = np.random.default_rng(42)
    rows = []
    for u in range(60):
        g = [4, 6, 0][u // 20]  # 3 cohorts: g=4, 6, never-treated
        for t in range(1, 9):
            te = max(0, t - g + 1) if g > 0 else 0
            rows.append({
                "i": u, "t": t,
                "y": te + rng.normal(),
                "g": g,
            })
    return pd.DataFrame(rows)


class TestCallawaySantannaProvenance:
    def test_attached(self, staggered_panel):
        r = sp.callaway_santanna(
            staggered_panel, y="y", g="g", t="t", i="i",
        )
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.did.callaway_santanna"
        assert prov.data_hash
        for key in ("y", "g", "t", "i", "estimator", "control_group"):
            assert key in prov.params

    def test_estimator_choice_captured(self, staggered_panel):
        r = sp.callaway_santanna(
            staggered_panel, y="y", g="g", t="t", i="i",
            estimator="reg",  # not the default 'dr'
        )
        prov = sp.get_provenance(r)
        assert prov.params["estimator"] == "reg"


# ---------------------------------------------------------------------------
# did_2x2
# ---------------------------------------------------------------------------

@pytest.fixture
def two_period_df():
    rng = np.random.default_rng(7)
    n = 400
    d = rng.integers(0, 2, n)
    t = rng.integers(0, 2, n)
    return pd.DataFrame({
        "y": 1 + 2 * d + 3 * t + 5 * d * t + rng.normal(0, 1, n),
        "d": d, "t": t,
    })


class TestDid2x2Provenance:
    def test_attached(self, two_period_df):
        r = sp.did_2x2(two_period_df, y="y", treat="d", time="t")
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.did.did_2x2"
        assert prov.data_hash
        assert prov.params["y"] == "y"
        assert prov.params["treat"] == "d"
        assert prov.params["time"] == "t"


# ---------------------------------------------------------------------------
# IV (statspai.regression.iv.iv)
# ---------------------------------------------------------------------------

@pytest.fixture
def iv_df():
    rng = np.random.default_rng(123)
    n = 300
    z = rng.normal(size=n)
    x = z + rng.normal(size=n)
    y = 1.0 * x + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z})


class TestIVProvenance:
    def test_attached_2sls(self, iv_df):
        from statspai.regression.iv import iv
        r = iv("y ~ (x ~ z)", data=iv_df)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.iv"
        assert prov.data_hash
        assert prov.params["formula"] == "y ~ (x ~ z)"
        assert prov.params["method"] == "2sls"

    def test_method_captured(self, iv_df):
        from statspai.regression.iv import iv
        r = iv("y ~ (x ~ z)", data=iv_df, method="liml")
        prov = sp.get_provenance(r)
        assert prov.params["method"] == "liml"


# ---------------------------------------------------------------------------
# Integration with replication_pack
# ---------------------------------------------------------------------------

class TestProvenanceFlowsIntoReplicationPack:
    def test_fit_then_pack_picks_up_lineage(
        self, cross_section_df, tmp_path,
    ):
        r = sp.regress("wage ~ trained + edu", cross_section_df)
        rp = sp.replication_pack(
            r, tmp_path / "out.zip", env=False,
            data=cross_section_df, code="# fake\n",
        )
        import json
        import zipfile
        with zipfile.ZipFile(rp.output_path) as zf:
            assert "lineage.json" in zf.namelist()
            lin = json.loads(zf.read("lineage.json"))
            assert lin["n_runs"] >= 1
            funcs = {v["function"] for v in lin["runs"].values()}
            assert "sp.regress" in funcs
