"""Stata parity for the panel tools ``sp.xtsum``, ``sp.xtserial``,
``sp.xtoverid`` and ``sp.xthtaylor`` (2026-11-05).

The reference numbers are real Stata 18 MP output on one committed panel
(``_fixtures/panel_xt_tools.csv``: 120 panels, half balanced and half with
gaps, AR(1) disturbances, a cluster variable nesting the panels), written
by ``_fixtures/_generate_panel_xt_tools_stata.do``: Stata's ``xtsum`` and
``xthtaylor``, ``xtserial`` 1.1.0 (Stata Journal st0039) and ``xtoverid``
2.1.8 (SSC), the last two run as black boxes.

Tolerances
----------
Every quantity is closed-form given the data (no iteration, no
quadrature), so the comparisons are at 1e-9 relative; the observed
agreement is reported in ``REFERENCES.md``. Degrees of freedom and counts
are compared exactly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def G():
    table = pd.read_csv(
        FIX / "panel_xt_tools_Stata.csv", skipinitialspace=True, dtype=str
    )
    return {
        k: (v if k.startswith("meta.") else float(v))
        for k, v in zip(table["key"], table["value"])
    }


@pytest.fixture(scope="module")
def panel():
    # correctly rounded parsing, as Stata's import delimited: pandas' default
    # parser differs in the last bit on about a fifth of these values
    return pd.read_csv(FIX / "panel_xt_tools.csv", float_precision="round_trip")


def close(ours, stata, rtol=1e-9, atol=0.0):
    assert np.isclose(ours, stata, rtol=rtol, atol=atol), (ours, stata)


def test_fixture_records_its_provenance(G, panel):
    assert G["meta.stata_version"] == "18" and G["meta.edition"] == "MP"
    assert "version 1.1.0" in G["meta.xtserial"]
    assert "version 2.1.8" in G["meta.xtoverid"]
    # the CSV is the one the data script writes
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_xt_tools_data", FIX / "_generate_panel_xt_tools_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pd.testing.assert_frame_equal(mod.make_panel(), panel, check_exact=True)


# --------------------------------------------------------------------- xtsum
@pytest.mark.parametrize("v", ["y", "x1", "ti", "xm"])
def test_xtsum_every_stored_result(G, panel, v):
    t = sp.xtsum(panel, [v], id="id")
    k = f"xtsum.{v}"
    assert t.loc[(v, "overall"), "obs"] == G[f"{k}.N"]
    assert t.loc[(v, "between"), "obs"] == G[f"{k}.n"]
    close(t.loc[(v, "within"), "obs"], G[f"{k}.Tbar"])
    close(t.loc[(v, "overall"), "mean"], G[f"{k}.mean"])
    for comp, suffix in [("overall", ""), ("between", "_b"), ("within", "_w")]:
        for stat in ("sd", "min", "max"):
            want = G[f"{k}.{stat}{suffix}"]
            # a time-invariant variable has within sd 0 in both (Stata's is
            # rounding noise of the within transform, ours too)
            atol = 1e-12 if (v == "ti" and comp == "within" and stat == "sd") else 0
            close(t.loc[(v, comp), stat], want, atol=atol)


def test_xtsum_time_invariant_variable_has_no_within_variation(G, panel):
    t = sp.xtsum(panel, ["ti"], id="id")
    assert G["xtsum.ti.sd_w"] < 1e-12
    close(t.loc[("ti", "within"), "min"], t.loc[("ti", "overall"), "mean"])
    close(t.loc[("ti", "within"), "max"], t.loc[("ti", "overall"), "mean"])


# ------------------------------------------------------------------ xtserial
@pytest.mark.parametrize(
    "tag,xs",
    [("s1", ["x1", "w", "x2"]), ("s2", ["x1"]), ("s3", ["x1", "w", "x2", "ti"])],
)
def test_xtserial_matches_stata(G, panel, tag, xs):
    import warnings

    with warnings.catch_warnings():
        # ti differences to zero: omitted, as Stata's regress omits it
        warnings.simplefilter("ignore", UserWarning)
        out = sp.xtserial(panel, "y", xs, id="id", time="t")
    close(out["statistic"], G[f"xtserial.{tag}.F"])
    close(out["pvalue"], G[f"xtserial.{tag}.p"])
    close(out["rho"], G[f"xtserial.{tag}.corr"])
    assert out["df"] == (G[f"xtserial.{tag}.df"], G[f"xtserial.{tag}.df_r"])


def test_xtserial_balanced_subset(G, panel):
    out = sp.xtserial(
        panel[panel["bal"] == 1], "y", ["x1", "w", "x2"], id="id", time="t"
    )
    close(out["statistic"], G["xtserial.bal.F"])
    close(out["pvalue"], G["xtserial.bal.p"])
    close(out["rho"], G["xtserial.bal.corr"])


def test_xtserial_time_invariant_regressor_does_not_change_the_test(G):
    # the s3 specification adds ti, which first differences remove
    for s in ("F", "p", "corr"):
        assert G[f"xtserial.s3.{s}"] == G[f"xtserial.s1.{s}"]


# ------------------------------------------------------------------ xtoverid
@pytest.mark.parametrize(
    "tag,xs,cluster,sub",
    [
        ("rob", ["x1", "w", "x2", "ti"], None, False),
        ("clu", ["x1", "w", "x2", "ti"], "g", False),
        ("two", ["x1", "x2"], None, False),
        ("bal", ["x1", "w", "x2", "ti"], None, True),
    ],
)
def test_xtoverid_matches_stata(G, panel, tag, xs, cluster, sub):
    data = panel[panel["bal"] == 1] if sub else panel
    out = sp.xtoverid(data, "y", xs, id="id", cluster=cluster)
    close(out["statistic"], G[f"xtoverid.{tag}.j"])
    close(out["pvalue"], G[f"xtoverid.{tag}.p"])
    assert out["df"] == G[f"xtoverid.{tag}.df"]
    # the time-invariant regressor is kept in the model but not tested
    assert "ti" not in out["tested"]


# ----------------------------------------------------------------- xthtaylor
_F = "y ~ x1 + w + x2 + ti + zi"
_HT = {
    "conv": dict(endog=["x2", "zi"]),
    "rob": dict(endog=["x2", "zi"], vce="robust"),
    "clu": dict(endog=["x2", "zi"], cluster="g"),
    "endog3": dict(endog=["w", "x2", "zi"]),
    "am_conv": dict(endog=["x2", "zi"], method="amacurdy", time="t"),
    "am_rob": dict(endog=["x2", "zi"], method="amacurdy", time="t", vce="robust"),
    "am_clu": dict(endog=["x2", "zi"], method="amacurdy", time="t", cluster="g"),
}


@pytest.mark.parametrize("tag", list(_HT))
def test_xthtaylor_matches_stata(G, panel, tag):
    data = panel[panel["bal"] == 1] if tag.startswith("am") else panel
    fit = sp.xthtaylor(_F, data, id="id", **_HT[tag])
    k = f"ht.{tag}"
    for v in ("x1", "w", "x2", "ti", "zi", "_cons"):
        close(fit.params[v], G[f"{k}.b_{v}"])
        close(fit.std_errors[v], G[f"{k}.se_{v}"])
    close(fit.model_info["sigma_u"], G[f"{k}.sigma_u"])
    close(fit.model_info["sigma_e"], G[f"{k}.sigma_e"])
    close(fit.model_info["rho"], G[f"{k}.rho"])
    assert fit.data_info["nobs"] == G[f"{k}.N"]
    if f"{k}.chi2" in G and tag in ("conv", "endog3", "am_conv"):
        close(fit.diagnostics["Wald chi2"], G[f"{k}.chi2"])
