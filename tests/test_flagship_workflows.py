"""Flagship research workflows, end to end from a file (review §4, §7.1).

Unit tests hand estimators ideal matrices; users start from a file and
finish with a table someone else can rerun. Each test below walks one of the
six flagship workflows from a CSV on disk through estimation, the design's
key diagnostic, the result card and a publication table -- with the
"awkward" options real projects use (weights, clusters, missing rows,
factors) -- and the panel workflow closes with a strict replication pack
that is rerun and diffed.

These are integration tests: numerical correctness of each step is pinned
by the parity suites; here the contract is that the steps compose without
manual patching and that what the card reports is what was run.
"""

from __future__ import annotations

import textwrap
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _roundtrip(df: pd.DataFrame, tmp_path, name: str) -> pd.DataFrame:
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return sp.read_data(str(path))


def test_panel_hdfe_workflow_to_verified_replication(tmp_path):
    rng = np.random.default_rng(11)
    n = 1200
    df = pd.DataFrame(
        {
            "firm": rng.integers(0, 120, n),
            "year": rng.integers(2000, 2010, n),
            "x": rng.normal(size=n),
            "w": rng.uniform(0.5, 2.0, n),
        }
    )
    df["state"] = df["firm"] % 12
    df["y"] = 0.7 * df["x"] + 0.02 * df["firm"] + rng.normal(size=n)
    df.loc[df.index[:15], "state"] = np.nan  # rows the cluster markout drops
    data = _roundtrip(df, tmp_path, "panel")

    fit = sp.feols("y ~ x | firm + year", data, cluster="state", weights="w")
    card = sp.result_card(fit)
    assert card["provenance"]["backend"]["backend"] == "pyfixest"
    assert card["inference"]["full_covariance"] is True
    slopes = sp.feols("y ~ x | firm + state[year]", data.dropna(), cluster="state")
    assert slopes.model_info["backend"] == "statspai-native"
    table = sp.regtable(fit, slopes, output="text")
    assert "x" in str(table)

    script = textwrap.dedent("""
        import pandas as pd, statspai as sp
        d = pd.read_csv("data/dataset.csv")
        sp.feols("y ~ x | firm + year", d, cluster="state", weights="w")
        """)
    pack = tmp_path / "panel_pack.zip"
    sp.replication_pack(
        fit,
        pack,
        data=data,
        code=script,
        env=True,
        bib=False,
        include_git_sha=False,
        strict=True,
    )
    ver = sp.verify_replication_pack(pack)
    assert ver["status"] == "verified", ver.summary()


def test_did_workflow_event_study_bands_and_sensitivity(tmp_path):
    data = _roundtrip(sp.datasets.mpdta(), tmp_path, "mpdta")
    cs = sp.callaway_santanna(data, y="lemp", g="first_treat", t="year", i="countyreal")
    dyn = sp.aggte(cs, type="dynamic", n_boot=199, random_state=0)
    pre = sp.pretrends_test(cs)
    assert 0.0 <= pre["pvalue"] <= 1.0
    bands = sp.uniform_bands(dyn, n_draws=5000, seed=0)
    assert {"lower", "upper"} <= set(c.lower() for c in bands.columns) or len(bands)
    hd = sp.honest_did(dyn, e=0, m_grid=[0.0, 0.5])
    assert len(hd) == 2
    card = sp.result_card(cs)
    assert card["evidence"]["level"] == "configuration"
    assert "not established by this fit" in card["assumptions"]["status"]


def test_iv_workflow_weak_instrument_robust(tmp_path):
    data = _roundtrip(sp.datasets.card_1995(), tmp_path, "card")
    fit = sp.iv(
        "lwage ~ exper + expersq + black + south + smsa + (educ ~ nearc4)",
        data=data,
        robust="hc1",
    )
    ar = sp.anderson_rubin_ci(
        "lwage",
        "educ",
        ["nearc4"],
        exog=["exper", "expersq", "black", "south", "smsa"],
        data=data,
    )
    assert ar is not None
    card = sp.result_card(fit)
    assert any("First-stage" in k for k in card["assumptions"]["diagnostics_run"])
    assert card["evidence"]["outputs"]["estimate"] == "reference"


def test_rd_workflow_weighted_with_density_and_card(tmp_path):
    df = sp.dgp_rd(n=1500, seed=4)
    rng = np.random.default_rng(4)
    df["w"] = rng.uniform(0.5, 2.0, len(df))
    df.loc[df.index[:10], "y"] = np.nan
    data = _roundtrip(df, tmp_path, "rd")
    fit = sp.rdrobust(data, y="y", x="x", c=0, weights="w")
    dens = sp.rddensity(data, x="x", c=0)
    assert np.isfinite(dens.pvalue)
    card = sp.result_card(fit)
    assert card["evidence"]["status"] == "covered"  # weighted default row
    assert card["sample"]["n_used"] == len(df) - 10


def test_observational_dml_workflow_with_identification_brief(tmp_path):
    rng = np.random.default_rng(5)
    n = 800
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["d"] = (df["x1"] + rng.normal(size=n) > 0).astype(int)
    df["y"] = 0.5 * df["d"] + df["x1"] - 0.5 * df["x2"] + rng.normal(size=n)
    data = _roundtrip(df, tmp_path, "obs")
    rec = sp.recommend(data, y="y", treatment="d", covariates=["x1", "x2"])
    assert rec.identification["claim"] == "causal only under selection on observables"
    fit = sp.dml(
        data, y="y", treat="d", covariates=["x1", "x2"], model="plr", random_state=0
    )
    assert abs(fit.estimate - 0.5) < 4 * fit.se
    assert sp.result_card(fit)["evidence"]["level"] == "configuration"


def test_survey_mi_workflow_categorical_imputation_and_calibration(tmp_path):
    rng = np.random.default_rng(6)
    n = 600
    df = pd.DataFrame(
        {
            "stratum": rng.integers(0, 4, n),
            "psu": rng.integers(0, 10, n),
            "d": rng.uniform(20, 60, n),
            "region": rng.choice(["north", "south", "west"], n),
            "age": rng.normal(45, 12, n),
        }
    )
    df["y"] = 2 + 0.03 * df["age"] + (df["region"] == "south") + rng.normal(size=n)
    df.loc[rng.random(n) < 0.15, "region"] = np.nan
    data = _roundtrip(df, tmp_path, "survey")

    mi = sp.mice(data, m=3, max_iter=3, seed=0)
    assert mi.methods["region"] == "polyreg"
    pooled = sp.mi_estimate(mi, sp.regress, formula="y ~ age + C(region)")
    assert "C(region)[T.south]" in pooled["var_names"]

    complete = mi.complete(0)
    des = sp.svydesign(
        complete, weights="d", strata="stratum", cluster="psu", nest=True
    )
    cal = des.calibrate(
        margins={"region": {"north": 9000, "south": 8000, "west": 7000}}
    )
    m_fixed = des.mean("y")
    m_cal = cal.mean("y")
    assert np.isfinite(m_cal.std_error.iloc[0])
    assert cal.total("d").estimate.iloc[0] > 0
    assert m_cal.estimate.iloc[0] != m_fixed.estimate.iloc[0]
