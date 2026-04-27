"""Round-2 estimator provenance instrumentation.

Layered on top of the Phase 3 instrumentation (regress, callaway_santanna,
did_2x2, iv). This round closes the deferred ``sp.synth`` dispatcher
and adds 4 more high-value estimators:

- ``sp.synth`` — refactored into outer wrapper + ``_dispatch_synth_impl``
  so the 13-method dispatcher attaches provenance once on the way out.
- ``sp.did.did_imputation`` — Borusyak-Jaravel-Spiess (2024) imputation.
- ``sp.did.aggte`` — Callaway-Sant'Anna ATT(g, t) aggregation.
- ``sp.did.did_multiplegt`` — de Chaisemartin-D'Haultfoeuille (2020).
- ``sp.rd.rdrobust`` — local-polynomial RD with robust bias correction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# sp.synth dispatcher
# ---------------------------------------------------------------------------

@pytest.fixture
def scm_panel():
    """California-style synthetic-control DGP with one treated unit."""
    rng = np.random.default_rng(0)
    units = [f"state_{i}" for i in range(20)]
    years = list(range(1980, 2001))
    rows = []
    for u in units:
        is_treated = (u == "state_0")
        for y in years:
            post = int(y >= 1990 and is_treated)
            rows.append({
                "state": u, "year": y,
                "gdp": 100 + 5 * (y - 1980) + rng.normal(0, 3) - 8 * post,
            })
    return pd.DataFrame(rows)


class TestSynthProvenance:
    def test_default_method_attached(self, scm_panel):
        r = sp.synth(
            scm_panel, outcome="gdp", unit="state", time="year",
            treated_unit="state_0", treatment_time=1990, placebo=False,
        )
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.synth"
        assert prov.data_hash
        assert prov.params["outcome"] == "gdp"
        assert prov.params["unit"] == "state"
        assert prov.params["treatment_time"] == 1990

    def test_method_choice_captured(self, scm_panel):
        r = sp.synth(
            scm_panel, outcome="gdp", unit="state", time="year",
            treated_unit="state_0", treatment_time=1990,
            method="classic", placebo=False,
        )
        prov = sp.get_provenance(r)
        assert prov.params["method"] == "classic"

    def test_signature_unchanged(self, scm_panel):
        # Refactor must preserve full backward-compatibility.
        r = sp.synth(
            scm_panel, outcome="gdp", unit="state", time="year",
            treated_unit="state_0", treatment_time=1990, placebo=False,
        )
        # The CausalResult shape is unchanged.
        from statspai.core.results import CausalResult
        assert isinstance(r, CausalResult)
        assert hasattr(r, "estimate")
        assert hasattr(r, "se")
        assert hasattr(r, "ci")


# ---------------------------------------------------------------------------
# sp.did.did_imputation
# ---------------------------------------------------------------------------

@pytest.fixture
def staggered_panel_first_treat():
    rng = np.random.default_rng(42)
    rows = []
    for u in range(60):
        cohort = [4, 6, np.inf][u // 20]  # never-treated = inf
        for t in range(1, 9):
            te = max(0, t - cohort + 1) if np.isfinite(cohort) else 0
            rows.append({
                "i": u, "t": t,
                "y": te + rng.normal(),
                "ftreat": cohort,
            })
    return pd.DataFrame(rows)


class TestDidImputationProvenance:
    def test_attached(self, staggered_panel_first_treat):
        from statspai.did.did_imputation import did_imputation
        r = did_imputation(
            staggered_panel_first_treat,
            y="y", group="i", time="t", first_treat="ftreat",
        )
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.did.did_imputation"
        assert prov.data_hash
        for key in ("y", "group", "time", "first_treat"):
            assert key in prov.params


# ---------------------------------------------------------------------------
# sp.did.aggte (depends on callaway_santanna upstream provenance)
# ---------------------------------------------------------------------------

class TestAggteProvenance:
    def test_attached_with_upstream_link(self):
        rng = np.random.default_rng(0)
        rows = []
        for u in range(60):
            g = [4, 6, 0][u // 20]
            for t in range(1, 9):
                te = max(0, t - g + 1) if g > 0 else 0
                rows.append({"i": u, "t": t, "y": te + rng.normal(), "g": g})
        df = pd.DataFrame(rows)
        cs = sp.callaway_santanna(df, y="y", g="g", t="t", i="i")
        upstream_prov = sp.get_provenance(cs)
        assert upstream_prov is not None  # Phase 3.2 already covers this

        from statspai.did.aggte import aggte
        agg = aggte(cs, type="simple", bstrap=False)
        prov = sp.get_provenance(agg)
        assert prov is not None
        assert prov.function == "sp.did.aggte"
        assert prov.params["type"] == "simple"
        # Lineage links upstream + downstream.
        assert prov.params["upstream_run_id"] == upstream_prov.run_id
        assert prov.params["upstream_function"] == upstream_prov.function


# ---------------------------------------------------------------------------
# sp.did.did_multiplegt
# ---------------------------------------------------------------------------

@pytest.fixture
def multiplegt_panel():
    rng = np.random.default_rng(7)
    rows = []
    for u in range(40):
        treat_seq = [0, 0, 0, 1, 1, 0, 0]  # treatment can switch off
        for t, d in enumerate(treat_seq):
            rows.append({
                "i": u, "t": t,
                "y": 0.5 * d + rng.normal(),
                "d": d,
            })
    return pd.DataFrame(rows)


class TestDidMultiplegtProvenance:
    def test_attached(self, multiplegt_panel):
        from statspai.did.did_multiplegt import did_multiplegt
        r = did_multiplegt(
            multiplegt_panel,
            y="y", group="i", time="t", treatment="d",
            n_boot=20,  # small for speed
        )
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.did.did_multiplegt"
        assert prov.data_hash
        assert prov.params["treatment"] == "d"


# ---------------------------------------------------------------------------
# sp.rd.rdrobust
# ---------------------------------------------------------------------------

@pytest.fixture
def rd_df():
    rng = np.random.default_rng(1)
    n = 1000
    x = rng.uniform(-1, 1, size=n)
    treated = (x >= 0).astype(int)
    y = 1.0 * treated + 0.5 * x + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x": x})


class TestRdrobustProvenance:
    def test_attached(self, rd_df):
        from statspai.rd.rdrobust import rdrobust
        r = rdrobust(rd_df, y="y", x="x", c=0.0)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.rd.rdrobust"
        assert prov.data_hash
        assert prov.params["c"] == 0.0
        assert prov.params["kernel"] == "triangular"

    def test_kernel_choice_captured(self, rd_df):
        from statspai.rd.rdrobust import rdrobust
        r = rdrobust(rd_df, y="y", x="x", c=0.0, kernel="uniform")
        prov = sp.get_provenance(r)
        assert prov.params["kernel"] == "uniform"


# ---------------------------------------------------------------------------
# Integration: replication_pack picks up lineage from many estimators.
# ---------------------------------------------------------------------------

class TestMultiEstimatorLineage:
    def test_pack_collects_all_provenance(self, scm_panel, rd_df, tmp_path):
        # Run two different estimators, then pack them together.
        synth_r = sp.synth(
            scm_panel, outcome="gdp", unit="state", time="year",
            treated_unit="state_0", treatment_time=1990, placebo=False,
        )
        from statspai.rd.rdrobust import rdrobust
        rd_r = rdrobust(rd_df, y="y", x="x", c=0.0)

        # Pack both — replication_pack walks ``_provenance`` on each.
        rp = sp.replication_pack(
            [synth_r, rd_r],
            tmp_path / "multi.zip",
            data=scm_panel, env=False,
        )
        import json
        import zipfile
        with zipfile.ZipFile(rp.output_path) as zf:
            assert "lineage.json" in zf.namelist()
            lin = json.loads(zf.read("lineage.json"))
            assert lin["n_runs"] >= 2
            funcs = {v["function"] for v in lin["runs"].values()}
            assert "sp.synth" in funcs
            assert "sp.rd.rdrobust" in funcs
