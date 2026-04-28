"""Round-3 estimator provenance instrumentation.

Layered on Phases 3+4 (9 estimators). This round adds 12 more:

DiD long-tail (10):
- ``sp.cic`` — Athey-Imbens (2006) Changes-in-Changes.
- ``sp.cohort_anchored_event_study`` — staggered-robust ES (arXiv:2509.01829).
- ``sp.design_robust_event_study`` — orthogonalised TWFE (arXiv:2601.18801).
- ``sp.gardner_did`` / ``sp.did_2stage`` — Gardner (2021) two-stage.
- ``sp.harvest_did`` — Borusyak et al. (2025) harvesting.
- ``sp.did_misclassified`` — misclassification + anticipation.
- ``sp.stacked_did`` — Cengiz et al. (2019) stacked.
- ``sp.wooldridge_did`` — Wooldridge (2021) ETWFE.
- ``sp.etwfe`` — wrapper-pattern dispatcher (4 internal branches).
- ``sp.drdid`` — Sant'Anna-Zhao (2020) doubly robust.

RD (2):
- ``sp.rd_honest`` — Armstrong-Kolesar (2018, 2020) honest CIs.
- ``sp.rkd`` — Card et al. (2015) Regression Kink Design.

Total provenance coverage after this round: **21/925**.
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
def staggered_panel_df():
    """Balanced staggered DiD panel with 3 cohorts."""
    rng = np.random.default_rng(42)
    rows = []
    for u in range(60):
        cohort = [4, 6, 0][u // 20]
        for t in range(1, 9):
            te = max(0, t - cohort + 1) if cohort > 0 else 0
            rows.append({
                "i": u, "t": t,
                "y": te + rng.normal(),
                "g": cohort,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def two_period_df():
    """Classic 2x2 DiD frame for cic / drdid."""
    rng = np.random.default_rng(7)
    rows = []
    for u in range(150):
        treat = u >= 75
        for t in (0, 1):
            y = (1.0 + 0.5 * treat + 0.3 * t + 1.2 * treat * t
                 + rng.normal())
            rows.append({"i": u, "t": t, "treat": int(treat), "y": y})
    return pd.DataFrame(rows)


@pytest.fixture
def event_study_panel_df():
    """Balanced panel for event-study estimators (Bartik-style staggered).

    Uses (id, period) layout. ``cohort`` is the first-treatment period
    (0 for never-treated); ``treat_bin`` is the binary 0/1 indicator;
    ``ftreat`` is the cohort label with ``np.nan`` for never-treated
    (the format ``did_imputation`` expects).
    """
    rng = np.random.default_rng(11)
    rows = []
    for u in range(120):
        cohort = [3, 5, 0][u // 40]  # 3 cohorts, 40 each
        for t in range(1, 9):
            d = int(cohort > 0 and t >= cohort)
            te = max(0, t - cohort + 1) if cohort > 0 else 0
            rows.append({
                "id": u, "period": t,
                "y": te + rng.normal(scale=0.5),
                "cohort": cohort,
                "treat_bin": d,
                "ftreat": cohort if cohort > 0 else np.nan,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def rd_df():
    """Sharp RD with discontinuity at x=0."""
    rng = np.random.default_rng(1)
    n = 800
    x = rng.uniform(-1, 1, size=n)
    treated = (x >= 0).astype(int)
    y = 1.0 * treated + 0.5 * x + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x": x})


@pytest.fixture
def rkd_df():
    """Regression-kink DGP: slope changes at x=0."""
    rng = np.random.default_rng(2)
    n = 1500
    x = rng.uniform(-1, 1, size=n)
    # Slope kink: dy/dx = 0.5 below, 1.5 above.
    y = np.where(x < 0, 0.5 * x, 1.5 * x) + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"y": y, "x": x})


# ---------------------------------------------------------------------------
# DiD long-tail
# ---------------------------------------------------------------------------

class TestCicProvenance:
    def test_attached(self, two_period_df):
        r = sp.cic(two_period_df, y="y", group="treat", time="t",
                   n_boot=20, n_grid=50)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.did.cic"
        assert prov.data_hash
        assert prov.params["y"] == "y"
        assert prov.params["group"] == "treat"
        assert prov.params["n_boot"] == 20


class TestCohortAnchoredProvenance:
    def test_attached(self, event_study_panel_df):
        # ``treat`` here is the cohort column (first-treatment period),
        # not a binary 0/1 indicator — matches the API.
        r = sp.cohort_anchored_event_study(
            event_study_panel_df, y="y", treat="cohort",
            time="period", id="id", leads=2, lags=2,
        )
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.did.cohort_anchored_event_study"
        assert prov.params["leads"] == 2 and prov.params["lags"] == 2


class TestDesignRobustProvenance:
    def test_attached(self, event_study_panel_df):
        # design_robust_event_study uses the same cohort-as-treat
        # convention as cohort_anchored.
        r = sp.design_robust_event_study(
            event_study_panel_df, y="y", treat="cohort",
            time="period", id="id", leads=2, lags=2,
        )
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.did.design_robust_event_study"


class TestGardnerProvenance:
    def test_attached(self, staggered_panel_df):
        r = sp.gardner_did(
            staggered_panel_df, y="y", group="i", time="t",
            first_treat="g",
        )
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.did.gardner_did"
        assert prov.params["first_treat"] == "g"

    def test_did_2stage_alias_same_provenance(self, staggered_panel_df):
        # did_2stage IS gardner_did (alias) — same provenance signature.
        r = sp.did_2stage(
            staggered_panel_df, y="y", group="i", time="t",
            first_treat="g",
        )
        prov = sp.get_provenance(r)
        assert prov.function == "sp.did.gardner_did"


class TestHarvestProvenance:
    def test_attached(self, staggered_panel_df):
        # harvest_did needs first_treat-style cohort; use g column.
        r = sp.harvest_did(
            staggered_panel_df,
            unit="i", time="t", outcome="y",
            cohort="g", never_value=0,
            horizons=[1, 2, 3],
        )
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.did.harvest_did"
        assert prov.params["horizons"] == [1, 2, 3]


class TestMisclassifiedProvenance:
    def test_attached(self, event_study_panel_df):
        # did_misclassified uses cohort-as-treat too (treats values as
        # first-treatment-period labels).
        r = sp.did_misclassified(
            event_study_panel_df, y="y", treat="cohort",
            time="period", id="id",
            pi_misclass=0.05,
        )
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.did.did_misclassified"
        assert prov.params["pi_misclass"] == 0.05


class TestStackedProvenance:
    def test_attached(self, staggered_panel_df):
        r = sp.stacked_did(
            staggered_panel_df, y="y", group="i", time="t",
            first_treat="g", window=(-2, 2),
        )
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.did.stacked_did"
        assert prov.params["window"] == [-2, 2]


class TestWooldridgeDidProvenance:
    def test_attached(self, staggered_panel_df):
        r = sp.wooldridge_did(
            staggered_panel_df, y="y", group="i", time="t",
            first_treat="g",
        )
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.did.wooldridge_did"


class TestEtwfeProvenance:
    def test_attached_dispatcher_route(self, staggered_panel_df):
        # When etwfe falls through to wooldridge_did (xvar=None,
        # panel=True, cgroup='notyet'), the inner provenance wins
        # (overwrite=False semantics, matching synth's cascade).
        r = sp.etwfe(
            staggered_panel_df, y="y", group="i", time="t",
            first_treat="g",
        )
        prov = sp.get_provenance(r)
        assert prov is not None
        # Either the inner (more specific) wooldridge_did record OR
        # the etwfe wrapper, depending on whether the inner was first.
        assert prov.function in {"sp.etwfe", "sp.did.wooldridge_did"}


class TestDrdidProvenance:
    def test_attached(self, two_period_df):
        # drdid expects 2x2 binary (group, time) layout.
        df = two_period_df.rename(columns={"treat": "g", "t": "tt"})
        r = sp.drdid(df, y="y", group="g", time="tt", n_boot=20)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.did.drdid"
        assert prov.params["method"] == "imp"


# ---------------------------------------------------------------------------
# RD
# ---------------------------------------------------------------------------

class TestRdHonestProvenance:
    def test_attached(self, rd_df):
        r = sp.rd_honest(rd_df, y="y", x="x", c=0.0)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.rd.rd_honest"
        assert prov.params["c"] == 0.0
        assert prov.params["kernel"] == "triangular"


class TestRkdProvenance:
    def test_attached(self, rkd_df):
        r = sp.rkd(rkd_df, y="y", x="x", c=0.0)
        prov = sp.get_provenance(r)
        assert prov is not None
        assert prov.function == "sp.rd.rkd"
        assert prov.params["c"] == 0.0


# ---------------------------------------------------------------------------
# Integration: every newly instrumented estimator's lineage flows
# into replication_pack.
# ---------------------------------------------------------------------------

class TestRound3LineageIntegration:
    def test_multi_estimator_pack(self, staggered_panel_df, rd_df, tmp_path):
        # Three estimators across two families → 3 distinct lineage runs.
        gardner = sp.gardner_did(
            staggered_panel_df, y="y", group="i", time="t",
            first_treat="g",
        )
        rd = sp.rd_honest(rd_df, y="y", x="x", c=0.0)
        rkd = sp.rkd(rd_df, y="y", x="x", c=0.0)

        rp = sp.replication_pack(
            [gardner, rd, rkd],
            tmp_path / "round3.zip",
            data=staggered_panel_df, env=False,
        )
        import json
        import zipfile
        with zipfile.ZipFile(rp.output_path) as zf:
            assert "lineage.json" in zf.namelist()
            lin = json.loads(zf.read("lineage.json"))
            assert lin["n_runs"] >= 3
            funcs = {v["function"] for v in lin["runs"].values()}
            assert "sp.did.gardner_did" in funcs
            assert "sp.rd.rd_honest" in funcs
            assert "sp.rd.rkd" in funcs
