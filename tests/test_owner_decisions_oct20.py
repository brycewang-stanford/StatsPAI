"""Owner decisions of 2026-10-20 (docs/dev/2026-10-20-owner-decisions.md).

Defaults that changed without changing a number, deprecations announced
with ``FutureWarning`` (numbers unchanged in this release), the public
reset of the spec-search counter, the opt-in summary text block of the MCP
server, and the cobalt convention for the SMD of a categorical level.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# 1. sp.causal_forest(n_jobs=-1) by default; the estimates do not move
# --------------------------------------------------------------------------


def test_causal_forest_default_threads_is_all_cores_and_changes_nothing():
    assert inspect.signature(sp.causal_forest).parameters["n_jobs"].default == -1
    assert inspect.signature(sp.CausalForest).parameters["n_jobs"].default == -1
    rng = np.random.default_rng(5)
    n = 400
    X = rng.normal(size=(n, 2))
    T = rng.binomial(1, 0.5, n)
    Y = X[:, 0] * T + X[:, 1] + rng.normal(size=n)
    default = sp.causal_forest(Y=Y, T=T, X=X, n_estimators=60, random_state=4)
    single = sp.causal_forest(Y=Y, T=T, X=X, n_estimators=60, random_state=4, n_jobs=1)
    assert default.n_jobs == -1 and single.n_jobs == 1
    np.testing.assert_array_equal(default.predict(), single.predict())
    a, b = default.average_treatment_effect(), single.average_treatment_effect()
    assert a["estimate"] == b["estimate"] and a["se"] == b["se"]


# --------------------------------------------------------------------------
# 2. sp.hdfe_ols: multi-way cluster_df default announced, numbers unchanged
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(11)
    firm = np.repeat(np.arange(40), 15)
    year = np.tile(np.arange(15), 40)
    x = rng.normal(size=firm.size)
    y = 0.5 * x + 0.3 * firm / 40 + 0.1 * year + rng.normal(size=firm.size)
    return pd.DataFrame({"y": y, "x": x, "firm": firm, "year": year})


def test_hdfe_ols_multiway_default_warns_and_keeps_per_term(panel):
    f = "y ~ x | firm + year"
    with pytest.warns(FutureWarning, match="cluster_df="):
        default = sp.hdfe_ols(f, data=panel, cluster=["firm", "year"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        per_term = sp.hdfe_ols(
            f, data=panel, cluster=["firm", "year"], cluster_df="per_term"
        )
        g_min = sp.hdfe_ols(f, data=panel, cluster=["firm", "year"], cluster_df="min")
        sp.hdfe_ols(f, data=panel, cluster="firm")  # one-way: no warning
    # The implicit default is exactly today's per-term factor.
    assert default.std_errors["x"] == per_term.std_errors["x"]
    assert default.cluster_info["cluster_df"] == "per_term"
    # And 'min' is a different documented convention (reghdfe / fixest):
    # every term carries G_min/(G_min-1), here G_min = 15 (years).
    assert g_min.cluster_info["cluster_df"] == "min"


# --------------------------------------------------------------------------
# 3. sp.sun_abraham: headline deprecation, numbers unchanged
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def staggered():
    return sp.dgp_did(n_units=80, n_periods=8, staggered=True, seed=3)


def test_sun_abraham_without_aggregation_warns_and_is_event_time(staggered):
    # crit_dist is named so that only the aggregation default is under test
    kw = dict(y="y", g="first_treat", t="time", i="unit", crit_dist="normal")
    with pytest.warns(FutureWarning, match="aggregation="):
        implicit = sp.sun_abraham(staggered, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        explicit = sp.sun_abraham(staggered, aggregation="event_time", **kw)
        fixest = sp.sun_abraham(staggered, aggregation="fixest_att", **kw)
    assert implicit.estimate == explicit.estimate
    assert implicit.se == explicit.se
    assert implicit.model_info["summary_aggregation"] == "event_time"
    # The future headline is already reported alongside, identically.
    assert fixest.estimate == implicit.model_info["att_fixest_att"]


def test_sp_did_forwards_the_missing_aggregation(staggered):
    with pytest.warns(FutureWarning, match="aggregation="):
        sp.did(
            staggered,
            y="y",
            treat="first_treat",
            time="time",
            id="unit",
            method="sun_abraham",
        )


# --------------------------------------------------------------------------
# 4. sp.bcf: rerouting announced; bcf_ordinal / factor_exposure untouched
# --------------------------------------------------------------------------


def _bcf_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    d = rng.binomial(1, 0.5, n)
    y = 1.5 * d + X[:, 0] + rng.normal(size=n)
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["y"], df["d"] = y, d
    return df


def test_bcf_announces_rerouting_and_keeps_its_numbers():
    from statspai.bcf.bcf import _bcf_fit

    df = _bcf_frame()
    kw = dict(y="y", treat="d", covariates=["x1", "x2"], n_bootstrap=5)
    with pytest.warns(FutureWarning, match="bart_causal"):
        pub = sp.bcf(df, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = _bcf_fit(df, **kw)
    assert pub.estimate == direct.estimate and pub.se == direct.se


def test_bcf_ordinal_does_not_emit_the_bcf_rerouting_warning():
    df = _bcf_frame(n=240, seed=1)
    df["dose"] = np.digitize(df["x2"], [-0.5, 0.5])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sp.bcf_ordinal(df, y="y", treat="dose", covariates=["x1"], n_bootstrap=3)
    assert not [w for w in rec if issubclass(w.category, FutureWarning)]


# --------------------------------------------------------------------------
# 5. matplotlib is a core dependency, still imported lazily
# --------------------------------------------------------------------------


def test_matplotlib_is_a_core_dependency_but_not_imported_at_import():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    meta = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    deps = meta["project"]["dependencies"]
    assert any(d.replace(" ", "").startswith("matplotlib>=3.5") for d in deps)
    plotting = meta["project"]["optional-dependencies"]["plotting"]
    assert any(d.startswith("seaborn") for d in plotting)
    code = "import sys, statspai; print('matplotlib' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


# --------------------------------------------------------------------------
# 6. sp.binscatter: classic + controls warns, the default stays classic
# --------------------------------------------------------------------------


def test_binscatter_classic_with_controls_warns_and_records_it():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(2)
    n = 500
    w = rng.normal(size=n)
    x = w + rng.normal(size=n)
    y = np.sin(x) + w + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({"y": y, "x": x, "w": w})
    assert inspect.signature(sp.binscatter).parameters["method"].default == "classic"
    with pytest.warns(sp.AssumptionWarning, match="method='binsreg'"):
        fig, _, bins = sp.binscatter(df, y="y", x="x", controls=["w"], n_bins=10)
    assert bins.attrs["diagnostics"]["suggest"] == "method='binsreg'"
    plt.close(fig)
    with warnings.catch_warnings():
        warnings.simplefilter("error", sp.AssumptionWarning)
        fig, _, bins = sp.binscatter(df, y="y", x="x", n_bins=10)
        plt.close(fig)
    assert "diagnostics" not in bins.attrs


# --------------------------------------------------------------------------
# 7. sp.reset_spec_counter
# --------------------------------------------------------------------------


def test_reset_spec_counter_forgets_what_was_counted():
    from statspai.core._spec_counter import specification_count

    rng = np.random.default_rng(9)
    df = pd.DataFrame(rng.normal(size=(120, 4)), columns=["yy", "a", "b", "c"])
    df["zz"] = rng.normal(size=120)
    sp.reset_spec_counter()
    for rhs in ("a", "a + b", "a + b + c"):
        sp.regress(f"yy ~ {rhs}", data=df)
    sp.regress("zz ~ a", data=df)
    assert specification_count(df, "yy") == 3
    out = sp.reset_spec_counter("yy")
    assert out == {"outcome": "yy", "families_cleared": 1, "specifications_cleared": 3}
    assert specification_count(df, "yy") == 0
    assert specification_count(df, "zz") == 1  # another outcome is kept
    assert sp.reset_spec_counter()["specifications_cleared"] == 1
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.reset_spec_counter(outcome=3)
    assert "reset_spec_counter" in sp.list_functions()


def test_reset_spec_counter_is_callable_over_mcp():
    from statspai.agent import mcp_server as m

    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "reset_spec_counter", "arguments": {}},
    }
    res = json.loads(m.handle_request(json.dumps(req)))["result"]
    assert not res["isError"]
    assert res["structuredContent"]["specifications_cleared"] >= 0


# --------------------------------------------------------------------------
# 8. MCP: opt-in summary text block
# --------------------------------------------------------------------------


def _mcp_regress():
    from statspai.agent import mcp_server as m

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=80), "z": rng.normal(size=80)})
    df["y"] = 1 + 2 * df["x"] + rng.normal(size=80)
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "regress",
            "arguments": {
                "formula": "y ~ x + z",
                "data_records": df.to_dict("records"),
            },
        },
    }
    return json.loads(m.handle_request(json.dumps(req)))["result"]


def test_mcp_text_block_default_is_full_json(monkeypatch):
    from statspai.agent import mcp_server as m

    monkeypatch.delenv("STATSPAI_MCP_TEXT", raising=False)
    m.set_text_mode(None)
    res = _mcp_regress()
    assert json.loads(res["content"][0]["text"]) == res["structuredContent"]


def test_mcp_summary_text_block_is_opt_in_and_keeps_structured(monkeypatch):
    from statspai.agent import mcp_server as m

    monkeypatch.setenv("STATSPAI_MCP_TEXT", "summary")
    m.set_text_mode(None)
    try:
        res = _mcp_regress()
    finally:
        monkeypatch.delenv("STATSPAI_MCP_TEXT")
        m.set_text_mode(None)
    text = res["content"][0]["text"]
    sc = res["structuredContent"]
    full = json.dumps(sc, separators=(",", ":"), allow_nan=False)
    assert "structuredContent" in text
    assert len(text) < len(full) / 5
    coef = sc["coefficients"]["x"]
    # The summary prints the headline numbers it was built from.
    assert f"{coef['estimate']:.6g}" in text
    assert f"{coef['std_error']:.6g}" in text
    assert f"N = {sc['n_obs']}" in text


def test_mcp_text_mode_rejects_unknown_values(monkeypatch):
    from statspai.agent import mcp_server as m

    with pytest.raises(ValueError, match="text mode"):
        m.set_text_mode("compact")
    m.set_text_mode(None)
    monkeypatch.setenv("STATSPAI_MCP_TEXT", "compact")
    with pytest.raises(ValueError, match="text mode"):
        m._text_mode()


# --------------------------------------------------------------------------
# 10. balance_table: categorical-level SMD on cobalt's binary convention
# --------------------------------------------------------------------------


def _cobalt_fixture():
    """RNG-free data, rebuilt identically in R (see the docstring below)."""
    n = 120
    i = np.arange(1, n + 1)
    u = np.mod(np.sin(i * 12.9898) * 43758.5453, 1.0)
    v = np.mod(np.sin(i * 78.233) * 12345.6789, 1.0)
    return pd.DataFrame(
        {
            "treat": (u < 0.4 + 0.3 * (v > 0.5)).astype(int),
            "region": np.array(["north", "south", "west"])[
                (np.floor(v * 7) % 3).astype(int)
            ],
            "grade": np.array(["A", "B"])[(np.floor(u * 11) % 2).astype(int)],
            "flag": (np.floor(v * 13) % 3) == 0,
            "age": np.round(20 + 40 * v + 5 * u, 3),
        }
    )


def test_balance_table_level_smd_equals_cobalt():
    """cobalt 4.6.2 (GPL >= 2, used as a black box: outputs only), R 4.5.2:
    n <- 120; i <- 1:n
    u <- (sin(i * 12.9898) * 43758.5453) %% 1
    v <- (sin(i * 78.233) * 12345.6789) %% 1
    treat <- as.integer(u < 0.4 + 0.3 * (v > 0.5))
    region <- factor(c("north", "south", "west")[1 + (floor(v * 7) %% 3)])
    grade <- c("A", "B")[1 + (floor(u * 11) %% 2)]
    flag <- (floor(v * 13) %% 3) == 0
    age <- round(20 + 40 * v + 5 * u, 3)
    bal.tab(treat ~ region + grade + flag + age, data = data.frame(...),
            binary = "std", continuous = "std", s.d.denom = "pooled")$Balance
    """
    cobalt = {
        "region: north": -0.183367477764663,
        "region: south": 0.19456333053765701,
        "region: west": 0.0024497335033944502,
        "grade: B": 0.67929403702547997,
        "flag: True": 0.12515198552995099,
        # numeric covariates keep the pooled sample-SD convention (unchanged)
        "age": 0.60005468329035505,
    }
    bt = sp.balance_table(
        _cobalt_fixture(),
        "treat",
        ["region", "grade", "flag", "age"],
        output="dataframe",
        fmt="%.17g",
    )
    for row, want in cobalt.items():
        # same formula on the same data: agreement to the 15 printed digits
        np.testing.assert_allclose(float(bt.loc[row, "SMD"]), want, rtol=1e-12)
    # The complementary level of a two-level variable is the mirror image.
    np.testing.assert_allclose(
        float(bt.loc["grade: A", "SMD"]), -float(bt.loc["grade: B", "SMD"]), rtol=1e-14
    )


def test_balance_table_level_smd_separating_level_uses_whole_sample_sd():
    """cobalt 4.6.2: treat = c(0,0,0,0,1,1), g = c(a,a,a,a,b,b) gives
    Diff.Un 2.12132034355964 for g_b: both arms constant, so the denominator
    is the whole sample's sqrt(p(1-p)) with p = 1/3."""
    df = pd.DataFrame({"treat": [0, 0, 0, 0, 1, 1], "g": list("aaaabb")})
    bt = sp.balance_table(df, "treat", ["g"], output="dataframe", fmt="%.17g")
    np.testing.assert_allclose(float(bt.loc["g: b", "SMD"]), 1 / np.sqrt(2 / 9))
    np.testing.assert_allclose(
        float(bt.loc["g: b", "SMD"]), 2.12132034355964, rtol=1e-13
    )


def test_internal_sun_abraham_wrappers_do_not_leak_the_futurewarning():
    """did_summary / did_analysis / auto_did call sun_abraham for the user,
    who cannot pass aggregation= through them, so they pin today's default
    and must not surface the headline FutureWarning."""
    import warnings

    import numpy as np
    import pandas as pd

    import statspai as sp

    rng = np.random.default_rng(0)
    n, T = 60, 8
    df = pd.DataFrame({"i": np.repeat(np.arange(n), T), "t": np.tile(np.arange(T), n)})
    df["g"] = np.repeat(rng.choice([0, 4, 6], n), T)
    df["y"] = rng.normal(size=len(df)) + ((df.t >= df.g) & (df.g > 0)) * 1.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sp.did_summary(df, y="y", time="t", first_treat="g", group="i")
    leaked = [
        x
        for x in w
        if issubclass(x.category, FutureWarning) and "aggregation" in str(x.message)
    ]
    assert leaked == []
