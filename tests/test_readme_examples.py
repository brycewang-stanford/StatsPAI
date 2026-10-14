"""Pin the beginner examples printed in README.md / README_CN.md.

The README once shipped outputs from a simulated Card replica, an RD call on
columns the dataset no longer had, and a DiD standard error that predated a
correctness fix. Nothing failed, because nothing read the README. These tests
run each example exactly as documented and require every non-``...`` line of
the matching ``text`` block to appear, in order, in the live output -- in both
language editions. Where a Track A golden exists for the same call, the
headline numbers are also held to the R reference.

The Prop 99 synthetic-control numbers are pinned separately in
``tests/test_synth_placebo_pvalue.py``; the summary layout is checked here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

import statspai as sp

ROOT = Path(__file__).resolve().parent.parent
READMES = [ROOT / "README.md", ROOT / "README_CN.md"]
R_RESULTS = ROOT / "tests" / "r_parity" / "results"

CARD_CONTROLS = ["exper", "expersq", "black", "south", "smsa"]


def _text_blocks(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8")
    return [
        block.splitlines()
        for block in re.findall(r"```text\n(.*?)\n```", text, flags=re.DOTALL)
    ]


def _block(path: Path, marker: str) -> list[str]:
    hits = [b for b in _text_blocks(path) if any(marker in line for line in b)]
    assert len(hits) == 1, f"{path.name}: expected one text block with {marker!r}"
    return hits[0]


def _assert_lines_in_order(documented: list[str], live: str, where: str) -> None:
    live_lines = [line.rstrip() for line in live.splitlines()]
    pos = 0
    for line in documented:
        line = line.rstrip()
        if line.strip() in {"", "..."}:
            continue
        try:
            pos = live_lines.index(line, pos) + 1
        except ValueError:
            raise AssertionError(
                f"{where}: README line not found (in order) in live output:\n"
                f"  {line!r}\n--- live output ---\n{live}"
            ) from None


def _golden(module: str, statistic: str) -> dict:
    rows = json.loads((R_RESULTS / f"{module}_R.json").read_text(encoding="utf-8"))
    (row,) = [r for r in rows["rows"] if r["statistic"] == statistic]
    return row


@pytest.fixture(scope="module")
def card():
    return sp.datasets.card_1995()


@pytest.fixture(scope="module")
def outputs(card):
    ols = sp.regress(
        "lwage ~ educ + exper + expersq + black + south + smsa",
        data=card,
        robust="hc1",
    )
    iv = sp.ivreg(
        "lwage ~ (educ ~ nearc4) + exper + expersq + black + south + smsa",
        data=card,
    )
    ar = sp.anderson_rubin_ci(
        y="lwage",
        endog="educ",
        instruments=["nearc4"],
        exog=CARD_CONTROLS,
        data=card,
    )
    gt = sp.callaway_santanna(
        data=sp.datasets.mpdta(),
        y="lemp",
        t="year",
        i="countyreal",
        g="first_treat",
    )
    overall = sp.aggte(gt, type="simple", bstrap=False)
    rd = sp.rdrobust(data=sp.datasets.lee_2008_senate(), y="y", x="x", c=0)
    sc = sp.synth(
        data=sp.datasets.california_prop99(),
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
    )
    cv = sp.cross_validate(
        card,
        "iv",
        y="lwage",
        endog=["educ"],
        instruments=["nearc4"],
        covariates=CARD_CONTROLS,
        engines=["statspai", "linearmodels"],
    )
    hiv = sp.datasets.thornton_hiv(complete_case=True)
    logit = sp.logit(
        "got ~ any + distvct + male + age", data=hiv, vce="cluster villnum"
    )
    ame = sp.margins(logit, variables=["any"])
    return {
        "hiv": hiv,
        "logit": logit,
        "ame": ame,
        "ols": ols,
        "iv": iv,
        "ar": ar,
        "overall": overall,
        "rd": rd,
        "sc": sc,
        "cv": cv,
    }


SUMMARY_BLOCKS = [
    ("ols", "Model: OLS"),
    ("iv", "Model: IV-2SLS"),
    ("ar", "Anderson-Rubin (AR)"),
    ("overall", "aggte[simple]"),
    ("rd", "Sharp RD Estimation"),
    ("sc", "Synthetic Control Method"),
    ("logit", "Model: Logit"),
]


@pytest.mark.parametrize("readme", READMES, ids=lambda p: p.name)
@pytest.mark.parametrize(
    "key,marker", SUMMARY_BLOCKS, ids=[k for k, _ in SUMMARY_BLOCKS]
)
def test_readme_summary_block_matches_live_output(readme, key, marker, outputs):
    _assert_lines_in_order(
        _block(readme, marker), outputs[key].summary(), f"{readme.name}:{key}"
    )


@pytest.mark.parametrize("readme", READMES, ids=lambda p: p.name)
def test_readme_cross_validate_rows_match(readme, outputs):
    documented = _block(readme, "VERDICT:")
    engine_rows = [
        line
        for line in documented
        if line.startswith(("Engine ", "statspai ", "linearmodels "))
    ]
    assert len(engine_rows) == 3
    _assert_lines_in_order(engine_rows, outputs["cv"].summary(), readme.name)


def test_readme_did_example_matches_r_did_golden(outputs):
    ref = _golden("04_csdid", "simple_ATT")
    # Track A pins R did::att_gt + aggte on these bytes at rel <= 1e-6.
    assert outputs["overall"].estimate == pytest.approx(ref["estimate"], rel=1e-6)
    assert outputs["overall"].se == pytest.approx(ref["se"], rel=1e-6)


def test_readme_rd_example_matches_r_rdrobust_golden(outputs):
    rd = outputs["rd"]
    for label in ("conventional", "robust"):
        ref = _golden("06_rd", f"default_{label}_est")
        got = rd.model_info[label]
        assert got["estimate"] == pytest.approx(ref["estimate"], rel=1e-6)
        assert got["se"] == pytest.approx(ref["se"], rel=1e-6)


def test_readme_card_ols_and_iv_headlines(outputs):
    ols, iv = outputs["ols"], outputs["iv"]
    assert ols.params["educ"] == pytest.approx(0.0740089942, abs=5e-10)
    assert ols.std_errors["educ"] == pytest.approx(0.0036420335, abs=5e-10)
    # Unadjusted 2SLS with the AER::ivreg / Stata `small` dof correction.
    assert iv.params["educ"] == pytest.approx(0.1322888400, abs=5e-10)
    assert iv.std_errors["educ"] == pytest.approx(0.0492332361, abs=5e-10)


def test_readme_lalonde_export_narrative():
    lal = sp.datasets.nsw_lalonde()
    naive = sp.regress("re78 ~ treat", data=lal, robust="hc1")
    full = sp.regress(
        "re78 ~ treat + age + educ + black + hispanic + married + nodegree"
        " + re74 + re75",
        data=lal,
        robust="hc1",
    )
    assert round(naive.params["treat"]) == -635
    assert round(full.params["treat"]) == 1548


@pytest.mark.parametrize("readme", READMES, ids=lambda p: p.name)
def test_readme_margins_block_matches_live_output(readme, outputs):
    _assert_lines_in_order(
        _block(readme, "dy/dx"), str(outputs["ame"].round(4)), readme.name
    )


def test_readme_logit_example_matches_stata(outputs):
    # Stata 18 MP on the same 2,825 rows (119 villages):
    #   logit got any distvct male age, vce(cluster villnum)
    #   margins, dydx(any) / test distvct / lincom any + male
    fit, ame = outputs["logit"], outputs["ame"]
    # Stata drops the rows with a missing cluster id / age itself.
    assert int(fit.data_info["nobs"]) == 2825
    assert fit.model_info["n_clusters"] == 119
    assert fit.model_info["n_missing_cluster_dropped"] == 4
    assert fit.params["any"] == pytest.approx(2.01780052, abs=5e-8)
    assert fit.std_errors["any"] == pytest.approx(0.09938486, abs=5e-8)
    assert fit.std_errors["distvct"] == pytest.approx(0.04077092, abs=5e-8)
    (row,) = ame.itertuples(index=False)
    assert row[1] == pytest.approx(0.3558466, abs=5e-7)
    assert row[2] == pytest.approx(0.0144976, abs=5e-7)
    assert fit.test("distvct = 0")["statistic"] == pytest.approx(17.31381895, rel=1e-6)
    lc = fit.lincom("any + male")
    assert lc["estimate"] == pytest.approx(1.96477656, abs=5e-8)
    assert lc["se"] == pytest.approx(0.14120214, abs=5e-8)


def test_readme_stata_runner_reproduces_margins(outputs):
    ame = sp.stata(
        "logit got any distvct male age, vce(cluster villnum)\n" "margins, dydx(any)",
        data=outputs["hiv"],
    )
    pd.testing.assert_frame_equal(ame, outputs["ame"])


def test_readme_house_style_aliases_are_the_same_call():
    mp = sp.datasets.mpdta()
    native = sp.callaway_santanna(
        data=mp, y="lemp", t="year", i="countyreal", g="first_treat"
    )
    alias = sp.callaway_santanna(
        data=mp, y="lemp", time="year", id="countyreal", first_treat="first_treat"
    )
    a = sp.aggte(native, type="simple", bstrap=False)
    b = sp.aggte(alias, type="simple", bstrap=False)
    assert (a.estimate, a.se) == (b.estimate, b.se)
    with pytest.raises(TypeError, match="did you mean 'running'"):
        sp.rdrobust(data=sp.datasets.lee_2008_senate(), y="y", runing="x", cutoff=0)
