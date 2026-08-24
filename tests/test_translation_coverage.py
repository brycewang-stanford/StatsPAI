"""The Stata/R migration on-ramps are reachable from Python and self-describing.

These translators were previously MCP-only; this suite locks in that
``sp.from_stata`` / ``sp.from_r`` / ``sp.translation_coverage`` are part of the
Python surface, and that the coverage matrix is introspected from the live
dispatch tables (so it cannot drift from what the translators actually do).
"""

from __future__ import annotations

import pytest

import statspai as sp


def test_translators_exposed_in_python_namespace():
    for name in ("from_stata", "from_r", "translation_coverage"):
        assert callable(getattr(sp, name, None)), f"sp.{name} not exposed"
        assert name in sp.list_functions(), f"sp.{name} not registered"


def test_from_stata_round_trip():
    out = sp.from_stata("reghdfe y x, absorb(id year) vce(cluster id)")
    assert out["ok"]
    # HDFE commands target the real, callable sp.feols (there is no sp.fixest
    # callable — it is a package), with FE folded into the pyfixest formula.
    assert out["tool"] == "feols"
    assert out["python_code"].startswith("sp.feols")
    assert out["arguments"]["fml"] == "y ~ x | id + year"
    # unrecognized command fails softly, with suggestions, not an exception
    bad = sp.from_stata("definitelynotacommand y x")
    assert bad["ok"] is False


def test_from_r_round_trip():
    out = sp.from_r("feols(y ~ x | id, data = df)")
    assert out["ok"]
    assert out["tool"] == "feols"
    assert out["python_code"].startswith("sp.feols")
    assert out["arguments"]["fml"] == "y ~ x | id"


def test_coverage_matrix_is_introspected_from_live_tables():
    cov = sp.translation_coverage()
    assert cov["summary"]["n_stata_commands"] >= 30
    assert cov["summary"]["n_r_functions"] >= 9
    # reghdfe really maps to sp.feols (parsed from the handler source)
    reghdfe = next(r for r in cov["stata"] if r["command"] == "reghdfe")
    assert "sp.feols" in reghdfe["targets"]
    # limitations are part of the contract
    assert any("panel" in lim.lower() for lim in cov["limitations"])


def test_coverage_markdown_renders():
    md = sp.translation_coverage(fmt="markdown")
    assert isinstance(md, str)
    assert "Stata" in md and "reghdfe" in md


# --------------------------------------------------------------------------- #
#  Executability — a translated payload must actually RUN, not just look right
# --------------------------------------------------------------------------- #
#
# A migration on-ramp that emits a plausible-but-unrunnable payload is worse
# than none: the user pastes their R/Stata command, gets a translation, runs it,
# and hits an error. This was real — the HDFE handlers emitted tool="fixest" /
# sp.fixest(formula=..., fe=[...]), but sp.fixest is a package (not callable),
# "fixest" is not a dispatchable tool, and feols wants the FE inside a pyfixest
# formula (fml=), not a separate fe= list. Every one of those translations was a
# dead on-ramp. These tests execute the emitted (tool, arguments) against
# synthetic data and assert it produces a real result.


def _exec_df():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 600
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "id": rng.integers(0, 20, n),
            "year": rng.integers(0, 8, n),
            "d": rng.normal(size=n),
            "z": rng.normal(size=n),
        }
    )


def _assert_executes(payload, df, label):
    import warnings

    from statspai.agent.tools import execute_tool

    assert payload["ok"], f"{label}: translation did not succeed: {payload}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = execute_tool(
            payload["tool"], payload["arguments"], data=df, detail="minimal"
        )
    err = res.get("error") or res.get("error_kind")
    assert not err, (
        f"{label}: translated payload sp.{payload['tool']}"
        f"({payload['arguments']}) did NOT execute — {err}: "
        f"{res.get('message') or res.get('error')}"
    )


def test_from_stata_hdfe_translations_execute():
    pytest.importorskip(
        "pyfixest", reason="sp.feols is backed by the optional [fixest] extra"
    )
    df = _exec_df()
    for cmd in (
        "reghdfe y x x2, absorb(id year) vce(cluster id)",
        "xtreg y x, fe i(id) vce(cluster id)",
        "ivreghdfe y x (d = z), absorb(id) cluster(id)",
    ):
        _assert_executes(sp.from_stata(cmd), df, f"from_stata: {cmd}")


def test_from_r_feols_translations_execute():
    pytest.importorskip(
        "pyfixest", reason="sp.feols is backed by the optional [fixest] extra"
    )
    df = _exec_df()
    for cmd in (
        "feols(y ~ x | id, data = df, cluster = ~id)",
        "feols(y ~ x | id + year, data = df)",
        "feols(y ~ 1 | id | d ~ z, data = df)",
        "feols(y ~ x, data = df)",
    ):
        _assert_executes(sp.from_r(cmd), df, f"from_r: {cmd}")
