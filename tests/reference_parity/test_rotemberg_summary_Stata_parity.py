"""``sp.rotemberg_summary`` = the Goldsmith-Pinkham, Sorkin & Swift
Rotemberg-weight summary table.

The China Syndrome replication (AER 2013) needed GPSS's summary of the
Bartik instrument -- weights, per-industry estimates, first-stage F, AR
intervals, by-year and by-sign breakdowns -- which ``sp.bartik`` did not
produce.

* Synthetic two-period panel (``_fixtures/rotemberg_panel.csv``): GPSS's own
  code (``bartik_weight`` + ``ch_weak``, the steps of their
  ``make_rotemberg_summary_ADH.do``; ``_generate_rotemberg_summary_Stata.do``).
  Industry aggregates, panels A-C and E to 1e-12, panel D's top five and AR
  intervals exactly.
* The published ADH table (GPSS 2020), when the GPSS replication inputs are
  available locally (``STATSPAI_GPSS_ADH_DIR`` pointing at the directory with
  ``ADHdata_AKM.csv``, ``Lshares.dta``, ``shocks.dta``): every printed number.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def synthetic():
    ref = json.loads(
        (_FIX / "rotemberg_summary_Stata.json").read_text(encoding="utf-8")
    )
    panel = pd.read_csv(_FIX / "rotemberg_panel.csv")
    sh = pd.read_csv(_FIX / "rotemberg_shocks.csv")
    shares = [c for c in panel.columns if c.startswith("sh")]
    shocks = pd.DataFrame(
        {"period": sh.year, "industry": "sh" + sh.ind.astype(str), "g": sh.g}
    )
    out = sp.rotemberg_summary(
        panel,
        y="y",
        x="x",
        shares=shares,
        shocks=shocks,
        time="year",
        covariates=["c1", "c2", "t2"],
        weights="w",
        cluster="unit",
    )
    return out, ref


def test_industry_aggregates(synthetic):
    out, ref = synthetic
    cols = ["ind", "alpha", "g", "beta", "F", "share", "share_sd"]
    r = pd.DataFrame(ref["industries"], columns=cols)
    r.index = "sh" + r["ind"].astype(int).astype(str)
    ind = out["industries"]
    assert list(ind.index) == list(r.index)  # both sorted by alpha
    for c in cols[1:]:
        np.testing.assert_allclose(ind[c], r[c], rtol=1e-11, atol=1e-13)


def test_panels(synthetic):
    out, ref = synthetic
    np.testing.assert_allclose(
        out["panel_a"].to_numpy().ravel(), ref["A_neg"] + ref["A_pos"], atol=1e-13
    )
    np.testing.assert_allclose(out["panel_b"].to_numpy(), ref["B"], atol=1e-12)
    np.testing.assert_allclose(
        out["panel_c"].to_numpy().ravel(), ref["C_1990"] + ref["C_2000"], atol=1e-13
    )
    np.testing.assert_allclose(
        out["panel_e"].to_numpy().ravel(),
        ref["E_neg"] + ref["E_pos"],
        rtol=1e-11,
        atol=1e-13,
    )
    d = out["panel_d"]
    assert [i[2:] for i in d.index] == list(ref["D_ci"])
    for k, (lo, hi) in ref["D_ci"].items():
        assert d.loc["sh" + k, "ci_lower"] == pytest.approx(lo, abs=1e-9)
        assert d.loc["sh" + k, "ci_upper"] == pytest.approx(hi, abs=1e-9)
    # the alpha-weighted sum of the just-identified estimates is 2SLS
    iv = sp.iv(
        "y ~ c1 + c2 + t2 + (x ~ bartik)",
        data=_with_bartik(out),
        weights="w",
    )
    assert out["beta"] == pytest.approx(float(iv.params["x"]), rel=1e-10)


def _with_bartik(out):
    panel = pd.read_csv(_FIX / "rotemberg_panel.csv")
    cells = out["cells"]
    g = cells.set_index(["period", "industry"])["g"]
    panel["bartik"] = [
        sum(r[k] * g[(r["year"], k)] for k in cells["industry"].unique())
        for _, r in panel.iterrows()
    ]
    return panel


def test_validation(synthetic):
    panel = pd.read_csv(_FIX / "rotemberg_panel.csv")
    with pytest.raises(sp.MethodIncompatibility, match="shocks"):
        sp.rotemberg_summary(
            panel,
            y="y",
            x="x",
            shares=["sh101"],
            shocks=pd.DataFrame({"industry": ["sh101"]}),
            time="year",
        )
    with pytest.raises(sp.MethodIncompatibility, match="not found"):
        sp.rotemberg_summary(
            panel,
            y="y",
            x="nope",
            shares=["sh101"],
            shocks=pd.DataFrame({"industry": ["sh101"], "g": [1.0]}),
        )


_ADH = os.environ.get("STATSPAI_GPSS_ADH_DIR")


@pytest.mark.skipif(
    not _ADH, reason="GPSS ADH inputs not available (STATSPAI_GPSS_ADH_DIR)"
)
def test_published_adh_table():
    """GPSS (2020), Rotemberg summary for ADH: every printed number."""
    d = pathlib.Path(_ADH)
    a = pd.read_csv(d / "ADHdata_AKM.csv")
    a["year"] = 1990 + 10 * (a.t2.astype(str).str.upper() == "TRUE")
    a["t2"] = (a.year == 2000).astype(int)
    L = pd.read_stata(d / "Lshares.dta")
    shk = pd.read_stata(d / "shocks.dta")
    L = L[~L.sic87dd.isin([2141, 3761])]  # zero shocks in both years (GPSS)
    W = L.pivot_table(
        index=["czone", "year"], columns="sic87dd", values="ind_share", fill_value=0.0
    )
    W.columns = [f"s{int(c)}" for c in W.columns]
    df = a.merge(W.reset_index(), on=["czone", "year"]).dropna(subset=["czone"])
    reg = pd.get_dummies(df.division, prefix="reg", drop_first=True, dtype=float)
    df = pd.concat([df, reg], axis=1)
    ctrl = list(reg.columns) + [
        "l_sh_popedu_c",
        "l_sh_popfborn",
        "l_sh_empl_f",
        "l_sh_routine33",
        "l_task_outsource",
        "l_shind_manuf_cbp",
        "t2",
    ]
    shocks = pd.DataFrame(
        {
            "period": shk.year,
            "industry": "s" + shk.sic87dd.astype(int).astype(str),
            "g": shk.g,
        }
    )
    shocks = shocks[shocks.industry.isin(W.columns)]
    o = sp.rotemberg_summary(
        df,
        y="d_sh_empl_mfg",
        x="shock",
        shares=list(W.columns),
        shocks=shocks,
        time="year",
        covariates=ctrl,
        weights="weights",
        cluster="czone",
    )
    r3 = lambda v: round(float(v), 3)  # noqa: E731
    assert [r3(v) for v in o["panel_a"]["sum"]] == [-0.067, 1.067]
    assert [r3(v) for v in o["panel_a"]["share"]] == [0.059, 0.941]
    b = o["panel_b"]
    assert r3(b.loc["alpha", "g"]) == 0.430 and r3(b.loc["g", "beta"]) == -0.320
    assert r3(b.loc["F", "share_sd"]) == 0.229
    assert [r3(v) for v in o["panel_c"]["sum"]] == [0.017, 0.983]
    top = o["panel_d"]
    assert list(top.index) == ["s3571", "s3944", "s3651", "s3661", "s3577"]
    assert [r3(v) for v in top["alpha"]] == [0.183, 0.138, 0.085, 0.066, 0.060]
    assert [r3(v) for v in top["beta"]] == [-0.619, -0.126, 0.174, -0.315, -0.303]
    assert [r3(v) for v in top["share_pct"]] == [0.137, 0.044, 0.046, 0.100, 0.100]
    assert (top.loc["s3571", "ci_lower"], top.loc["s3571", "ci_upper"]) == (-1.5, -0.2)
    assert not top.loc["s3661", "ci_bounded"]  # "N/A"
    e = o["panel_e"]
    assert [r3(v) for v in e["alpha_weighted_sum"]] == [-0.014, -0.582]
    assert [r3(v) for v in e["mean_beta"]] == [-0.036, -1.170]
    assert o["beta"] == pytest.approx(-0.5963601, abs=5e-8)
