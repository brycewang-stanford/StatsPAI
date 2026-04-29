"""Tests for the second v0.9.17 wave (previously deferred items):

- LTMLE dynamic regime + survival LTMLE
- conformal counterfactual + weighted conformal + ITE interval
- PCMCI time-series causal discovery
- ML-enhanced partial-ID bounds
- MCP server end-to-end
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════════════
#  LTMLE: dynamic regimes + survival
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def longitudinal_panel():
    rng = np.random.default_rng(0)
    n = 400
    L0 = rng.normal(size=n)
    A0 = (rng.random(n) < 1 / (1 + np.exp(-0.4 * L0))).astype(int)
    L1 = 0.3 * L0 + 0.4 * A0 + rng.normal(size=n)
    A1 = (rng.random(n) < 1 / (1 + np.exp(-(0.2 * L0 + 0.3 * L1)))).astype(int)
    Y = 0.5 + 0.6 * A0 + 0.4 * A1 + 0.3 * L0 + 0.2 * L1 + rng.normal(size=n) * 0.3
    return pd.DataFrame({"L0": L0, "A0": A0, "L1": L1, "A1": A1, "Y": Y})


class TestLTMLEDynamic:

    def test_dynamic_regime_runs(self, longitudinal_panel):
        import statspai as sp

        def policy(k, hist):
            key = f"L{k}"
            return (hist[key] > 0).astype(int)

        r = sp.ltmle(
            longitudinal_panel, y="Y",
            treatments=["A0", "A1"],
            covariates_time=[["L0"], ["L1"]],
            regime_treated=policy, regime_control=[0, 0],
        )
        assert np.isfinite(r.ate)
        assert r.detail["regime_treated_callable"] is True
        assert r.detail["regime_control_callable"] is False

    def test_dynamic_wrong_length_raises(self, longitudinal_panel):
        import statspai as sp

        def bad_policy(k, hist):
            return np.ones(10, dtype=int)   # wrong length

        with pytest.raises(ValueError, match="length"):
            sp.ltmle(
                longitudinal_panel, y="Y",
                treatments=["A0", "A1"],
                covariates_time=[["L0"], ["L1"]],
                regime_treated=bad_policy,
            )

    def test_static_list_still_works(self, longitudinal_panel):
        import statspai as sp
        r = sp.ltmle(
            longitudinal_panel, y="Y",
            treatments=["A0", "A1"],
            covariates_time=[["L0"], ["L1"]],
            regime_treated=[1, 1], regime_control=[0, 0],
        )
        assert np.isfinite(r.ate)

    def test_constant_callable_matches_static(self, longitudinal_panel):
        """A callable that always returns the static vector must give
        the same ATE as passing that vector directly."""
        import statspai as sp
        n = len(longitudinal_panel)

        def constant_one(k, hist):
            return np.ones(n, dtype=int)

        def constant_zero(k, hist):
            return np.zeros(n, dtype=int)

        r_static = sp.ltmle(
            longitudinal_panel, y="Y",
            treatments=["A0", "A1"],
            covariates_time=[["L0"], ["L1"]],
            regime_treated=[1, 1], regime_control=[0, 0],
        )
        r_dyn = sp.ltmle(
            longitudinal_panel, y="Y",
            treatments=["A0", "A1"],
            covariates_time=[["L0"], ["L1"]],
            regime_treated=constant_one, regime_control=constant_zero,
        )
        assert abs(r_static.ate - r_dyn.ate) < 1e-8


@pytest.fixture
def survival_panel():
    rng = np.random.default_rng(0)
    n = 500
    L0 = rng.normal(size=n)
    A0 = rng.binomial(1, 1 / (1 + np.exp(-0.3 * L0)))
    h1 = 1 / (1 + np.exp(-(-1.5 - 0.5 * A0 + 0.3 * L0)))
    T1 = rng.binomial(1, h1)

    L1 = 0.4 * L0 + rng.normal(size=n)
    A1 = rng.binomial(1, 1 / (1 + np.exp(-0.2 * L1)))
    h2 = 1 / (1 + np.exp(-(-1.3 - 0.6 * A1 + 0.2 * L1)))
    T2 = np.where(T1 == 1, 0, rng.binomial(1, h2))

    L2 = 0.4 * L1 + rng.normal(size=n)
    A2 = rng.binomial(1, 1 / (1 + np.exp(-0.2 * L2)))
    h3 = 1 / (1 + np.exp(-(-1.0 - 0.7 * A2 + 0.1 * L2)))
    T3 = np.where((T1 + T2) > 0, 0, rng.binomial(1, h3))

    return pd.DataFrame({
        "L0": L0, "A0": A0, "T1": T1,
        "L1": L1, "A1": A1, "T2": T2,
        "L2": L2, "A2": A2, "T3": T3,
    })


class TestLTMLESurvival:

    def test_produces_monotone_survival_curves(self, survival_panel):
        import statspai as sp
        res = sp.ltmle_survival(
            survival_panel,
            event_indicators=["T1", "T2", "T3"],
            treatments=["A0", "A1", "A2"],
            covariates_time=[["L0"], ["L1"], ["L2"]],
        )
        # Survival must be non-increasing in time
        assert np.all(np.diff(res.survival_treated) <= 1e-8)
        assert np.all(np.diff(res.survival_control) <= 1e-8)

    def test_treatment_improves_survival(self, survival_panel):
        import statspai as sp
        res = sp.ltmle_survival(
            survival_panel,
            event_indicators=["T1", "T2", "T3"],
            treatments=["A0", "A1", "A2"],
            covariates_time=[["L0"], ["L1"], ["L2"]],
        )
        # DGP sets negative coefficients on treatment → lower hazard →
        # higher survival
        assert res.survival_treated[-1] > res.survival_control[-1]
        assert res.rmst_difference > 0

    def test_rmst_ci_reasonable(self, survival_panel):
        import statspai as sp
        res = sp.ltmle_survival(
            survival_panel,
            event_indicators=["T1", "T2", "T3"],
            treatments=["A0", "A1", "A2"],
            covariates_time=[["L0"], ["L1"], ["L2"]],
        )
        lo, hi = res.rmst_ci
        assert lo <= res.rmst_difference <= hi

    def test_exposed_at_top_level(self):
        import statspai as sp
        assert callable(sp.ltmle_survival)
        assert hasattr(sp, "LTMLESurvivalResult")

    def test_rmst_se_magnitude_sanity(self, survival_panel):
        """RMST SE should be on the same order as a quick bootstrap SE,
        not inflated by a factor of K."""
        import statspai as sp
        res = sp.ltmle_survival(
            survival_panel,
            event_indicators=["T1", "T2", "T3"],
            treatments=["A0", "A1", "A2"],
            covariates_time=[["L0"], ["L1"], ["L2"]],
        )
        rng = np.random.default_rng(1)
        boot_diffs = []
        for _ in range(40):
            idx = rng.integers(0, len(survival_panel), size=len(survival_panel))
            df_b = survival_panel.iloc[idx].reset_index(drop=True)
            rb = sp.ltmle_survival(
                df_b,
                event_indicators=["T1", "T2", "T3"],
                treatments=["A0", "A1", "A2"],
                covariates_time=[["L0"], ["L1"], ["L2"]],
            )
            boot_diffs.append(rb.rmst_difference)
        boot_se = float(np.std(boot_diffs, ddof=1))
        ratio = res.rmst_se / max(boot_se, 1e-6)
        # Allow the IC-based SE to be within 0.4–2.5× of bootstrap —
        # catches an ×K-style inflation while still being lenient to
        # the first-order IC approximation.
        assert 0.4 < ratio < 2.5, (
            f"rmst_se={res.rmst_se}, boot_se={boot_se}, ratio={ratio}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Conformal counterfactual + weighted conformal + ITE interval
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def het_effect_data():
    rng = np.random.default_rng(0)
    n = 800
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    T = rng.binomial(1, 1 / (1 + np.exp(-(0.3 * x1 - 0.2 * x2))))
    tau = 0.5 + x1   # heterogeneous effect
    Y = 0.3 * x1 + 0.2 * x2 + T * tau + rng.normal(size=n) * 0.5
    df = pd.DataFrame({"Y": Y, "T": T, "x1": x1, "x2": x2})
    return df, tau


class TestConformalCounterfactual:

    def test_intervals_are_ordered(self, het_effect_data):
        import statspai as sp
        df, _ = het_effect_data
        cf = sp.conformal_counterfactual(
            df, y="Y", treat="T", covariates=["x1", "x2"],
            alpha=0.1, random_state=1,
        )
        assert np.all(cf.lower_Y1 <= cf.upper_Y1)
        assert np.all(cf.lower_Y0 <= cf.upper_Y0)

    def test_ite_interval_covers_true_effect(self, het_effect_data):
        import statspai as sp
        df, tau_true = het_effect_data
        ite = sp.conformal_ite_interval(
            df, y="Y", treat="T", covariates=["x1", "x2"],
            alpha=0.1, random_state=1,
        )
        # Nested bound is conservative — coverage should exceed target
        covered = np.mean((tau_true >= ite.lower) & (tau_true <= ite.upper))
        assert covered >= 0.85

    def test_weighted_conformal_basic_coverage(self):
        """Standard (unweighted) conformal should hit the nominal rate."""
        import statspai as sp
        rng = np.random.default_rng(0)
        n = 500
        X_all = rng.normal(size=(n, 3))
        y_all = X_all[:, 0] + 0.5 * X_all[:, 1] + rng.normal(size=n) * 0.5
        # 60/20/20 split
        n_trn, n_cal = 300, 100
        X_tr, y_tr = X_all[:n_trn], y_all[:n_trn]
        X_cal, y_cal = X_all[n_trn:n_trn + n_cal], y_all[n_trn:n_trn + n_cal]
        X_te, y_te = X_all[n_trn + n_cal:], y_all[n_trn + n_cal:]
        lo, hi, _ = sp.weighted_conformal_prediction(
            X_tr, y_tr, X_cal, y_cal, X_te, alpha=0.1,
        )
        covered = np.mean((y_te >= lo) & (y_te <= hi))
        assert 0.80 <= covered <= 1.0

    def test_exposed_at_top_level(self):
        import statspai as sp
        for name in ("conformal_counterfactual", "conformal_ite_interval",
                     "weighted_conformal_prediction",
                     "ConformalCounterfactualResult", "ConformalITEResult"):
            assert hasattr(sp, name), f"sp.{name} missing"


# ═══════════════════════════════════════════════════════════════════════
#  PCMCI
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def lagged_ts():
    # DGP: X0→X1 lag 1, X1→X2 lag 1, X0→X2 lag 2
    rng = np.random.default_rng(0)
    T = 800
    X0 = rng.normal(size=T)
    X1 = np.zeros(T); X2 = np.zeros(T)
    for t in range(2, T):
        X1[t] = 0.7 * X0[t - 1] + 0.3 * rng.normal()
        X2[t] = 0.5 * X1[t - 1] + 0.4 * X0[t - 2] + 0.3 * rng.normal()
    return pd.DataFrame({"X0": X0, "X1": X1, "X2": X2})


class TestPCMCI:

    def test_recovers_true_links(self, lagged_ts):
        import statspai as sp
        res = sp.pcmci(lagged_ts, tau_max=3, pc_alpha=0.01)
        links = res.discovered_links()
        true_links = {
            ("X0", "X1", 1),
            ("X1", "X2", 1),
            ("X0", "X2", 2),
        }
        got = {
            (row.source, row.target, row.lag)
            for row in links.itertuples()
        }
        # Must find all 3 true links
        assert true_links.issubset(got), (
            f"missing links: {true_links - got}, got {got}"
        )

    def test_no_lag_zero_self_loops(self, lagged_ts):
        import statspai as sp
        res = sp.pcmci(lagged_ts, tau_max=2, pc_alpha=0.01)
        # lag=0 adjacency should be all False by design
        assert not res.adjacency[0].any()

    def test_partial_corr_pvalue_respects_conditioning(self):
        import statspai as sp
        rng = np.random.default_rng(0)
        n = 1000
        Z = rng.normal(size=n)
        X = 0.9 * Z + 0.1 * rng.normal(size=n)
        Y = 0.9 * Z + 0.1 * rng.normal(size=n)
        # Marginally X and Y are highly correlated through Z
        p_marg = sp.partial_corr_pvalue(X, Y)
        # Conditionally they should be independent
        p_cond = sp.partial_corr_pvalue(X, Y, Z.reshape(-1, 1))
        assert p_marg < 0.001
        assert p_cond > 0.05

    def test_exposed_at_top_level(self):
        import statspai as sp
        for name in ("pcmci", "PCMCIResult", "partial_corr_pvalue"):
            assert hasattr(sp, name), f"sp.{name} missing"

    def test_type_I_control_on_white_noise(self):
        """Independent white-noise series should produce few links at α=0.01."""
        import statspai as sp
        rng = np.random.default_rng(0)
        T = 600
        df = pd.DataFrame({
            f"X{i}": rng.normal(size=T) for i in range(3)
        })
        res = sp.pcmci(df, tau_max=3, pc_alpha=0.01, mci_alpha=0.01)
        n_links = int(res.adjacency.sum())
        # 3 × 3 × 3 = 27 possible lagged edges, expected false positives
        # under the null ≈ 27 × 0.01 ≈ 0.3. Allow up to 4 to avoid
        # flakiness from Fisher-z tail heaviness at small samples.
        assert n_links <= 4, f"too many false positives: {n_links}"


# ═══════════════════════════════════════════════════════════════════════
#  ML-enhanced partial-identification bounds
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def bounds_data():
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    T = rng.binomial(1, 1 / (1 + np.exp(-(0.3 * x1 - 0.2 * x2))))
    Y = 0.3 * x1 + 0.2 * x2 + 0.8 * T + rng.normal(size=n) * 0.3
    return pd.DataFrame({"Y": Y, "T": T, "x1": x1, "x2": x2})


class TestMLBounds:

    def test_interval_contains_manski_midpoint(self, bounds_data):
        import statspai as sp
        res = sp.ml_bounds(
            bounds_data, y="Y", treat="T", covariates=["x1", "x2"],
            y_min=-3, y_max=3, n_bootstrap=30, random_state=1,
        )
        # Adaptive bounds should fully contain the Manski interval's midpoint
        manski_mid = 0.5 * (res.manski_lower + res.manski_upper)
        assert res.adaptive_lower <= manski_mid <= res.adaptive_upper

    def test_width_matches_y_range(self, bounds_data):
        """Under bounded-outcome Manski the width equals y_max - y_min."""
        import statspai as sp
        res = sp.ml_bounds(
            bounds_data, y="Y", treat="T", covariates=["x1", "x2"],
            y_min=-3, y_max=3, n_bootstrap=0, random_state=1,
        )
        expected_width = 3.0 - (-3.0)
        actual_width = res.adaptive_upper - res.adaptive_lower
        assert abs(actual_width - expected_width) < 1e-4

    def test_tighter_y_range_gives_tighter_bounds(self, bounds_data):
        import statspai as sp
        wide = sp.ml_bounds(
            bounds_data, y="Y", treat="T", covariates=["x1", "x2"],
            y_min=-5, y_max=5, n_bootstrap=0, random_state=1,
        )
        tight = sp.ml_bounds(
            bounds_data, y="Y", treat="T", covariates=["x1", "x2"],
            y_min=-2, y_max=3, n_bootstrap=0, random_state=1,
        )
        w_wide = wide.adaptive_upper - wide.adaptive_lower
        w_tight = tight.adaptive_upper - tight.adaptive_lower
        assert w_tight < w_wide

    def test_gradient_boosting_learner(self, bounds_data):
        import statspai as sp
        res = sp.ml_bounds(
            bounds_data, y="Y", treat="T", covariates=["x1", "x2"],
            learner="gradient_boosting", n_bootstrap=0, random_state=1,
        )
        assert "Gradient" in res.learner

    def test_exposed_at_top_level(self):
        import statspai as sp
        assert hasattr(sp, "ml_bounds")
        assert hasattr(sp, "MLBoundsResult")

    def test_nobootstrap_collapses_ci(self, bounds_data):
        """With n_bootstrap=0 the bootstrap CIs must collapse to the
        adaptive plug-in (no width around the endpoint)."""
        import statspai as sp
        res = sp.ml_bounds(
            bounds_data, y="Y", treat="T", covariates=["x1", "x2"],
            y_min=-3, y_max=3, n_bootstrap=0, random_state=1,
        )
        assert res.lower_ci == (res.adaptive_lower, res.adaptive_lower)
        assert res.upper_ci == (res.adaptive_upper, res.adaptive_upper)
        # Band equals the plug-in
        assert abs(res.lower - res.adaptive_lower) < 1e-12
        assert abs(res.upper - res.adaptive_upper) < 1e-12


# ═══════════════════════════════════════════════════════════════════════
#  MCP server
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def mcp_csv():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "y": rng.normal(size=n),
        "d": rng.binomial(1, 0.5, n),
        "x1": rng.normal(size=n),
    })
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False,
    )
    df.to_csv(tmp.name, index=False)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


class TestMCPServer:

    def test_initialize_returns_server_info(self):
        import statspai as sp
        req = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        resp = json.loads(sp.agent.mcp_handle_request(req))
        assert resp["result"]["serverInfo"]["name"] == "statspai"
        assert resp["result"]["serverInfo"]["version"] == sp.agent.MCP_SERVER_VERSION
        # Capability flags declared
        assert "tools" in resp["result"]["capabilities"]
        assert "resources" in resp["result"]["capabilities"]

    def test_tools_list_includes_data_path(self):
        import statspai as sp
        from statspai.agent.mcp_server import _DATALESS_TOOLS
        req = json.dumps({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
        })
        resp = json.loads(sp.agent.mcp_handle_request(req))
        tools = resp["result"]["tools"]
        assert len(tools) > 0
        for t in tools:
            schema = t["inputSchema"]
            assert "data_path" in schema["properties"]
            # Dataless tools (honest_did, sensitivity) advertise
            # data_path as an OPTIONAL convenience but must not mark
            # it required — strict-schema clients would otherwise
            # refuse to dispatch the call without a CSV path the
            # estimator never reads. See mcp_server._DATALESS_TOOLS.
            if t["name"] in _DATALESS_TOOLS:
                assert "data_path" not in schema["required"]
            else:
                assert "data_path" in schema["required"]

    def test_tools_call_runs_regress(self, mcp_csv):
        import statspai as sp
        req = json.dumps({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {
                "name": "regress",
                "arguments": {
                    "data_path": mcp_csv,
                    "formula": "y ~ d + x1",
                },
            },
        })
        resp = json.loads(sp.agent.mcp_handle_request(req))
        assert resp["result"]["isError"] is False
        content = resp["result"]["content"][0]
        assert content["type"] == "text"
        # Body is JSON-parseable
        result_obj = json.loads(content["text"])
        assert "coefficients" in result_obj or "params" in result_obj

    def test_tools_call_unknown_tool_returns_error(self, mcp_csv):
        import statspai as sp
        req = json.dumps({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {
                "name": "definitely_not_a_tool",
                "arguments": {"data_path": mcp_csv},
            },
        })
        resp = json.loads(sp.agent.mcp_handle_request(req))
        # Either JSON-RPC error or an isError:True content block
        assert resp["result"]["isError"] is True

    def test_resources_list_and_read(self):
        import statspai as sp
        req = json.dumps({
            "jsonrpc": "2.0", "id": 5, "method": "resources/list",
            "params": {},
        })
        resp = json.loads(sp.agent.mcp_handle_request(req))
        uris = [r["uri"] for r in resp["result"]["resources"]]
        assert "statspai://catalog" in uris

        req2 = json.dumps({
            "jsonrpc": "2.0", "id": 6, "method": "resources/read",
            "params": {"uri": "statspai://catalog"},
        })
        resp2 = json.loads(sp.agent.mcp_handle_request(req2))
        text = resp2["result"]["contents"][0]["text"]
        assert "StatsPAI tool catalog" in text

    def test_unknown_method_returns_jsonrpc_error(self):
        import statspai as sp
        req = json.dumps({
            "jsonrpc": "2.0", "id": 7, "method": "nosuch", "params": {},
        })
        resp = json.loads(sp.agent.mcp_handle_request(req))
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    def test_parse_error_returns_jsonrpc_32700(self):
        import statspai as sp
        resp = json.loads(sp.agent.mcp_handle_request("not valid json"))
        assert resp["error"]["code"] == -32700

    def test_notification_receives_no_response(self):
        """A JSON-RPC notification (no id) must return None."""
        import statspai as sp
        req = json.dumps({
            "jsonrpc": "2.0", "method": "initialize", "params": {},
        })
        assert sp.agent.mcp_handle_request(req) is None

    def test_relative_data_path_returns_error(self):
        """A relative data_path must be rejected, not crash."""
        import statspai as sp
        req = json.dumps({
            "jsonrpc": "2.0", "id": 8, "method": "tools/call",
            "params": {
                "name": "regress",
                "arguments": {
                    "data_path": "relative/path/data.csv",
                    "formula": "y ~ x",
                },
            },
        })
        resp = json.loads(sp.agent.mcp_handle_request(req))
        # Either a JSON-RPC error or a content block flagged isError=True
        assert ("error" in resp) or (resp.get("result", {}).get("isError") is True)
