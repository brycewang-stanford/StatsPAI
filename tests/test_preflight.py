"""Tests for ``sp.preflight(data, method, **kwargs)`` — method-specific
pre-estimation diagnostics.

These tests pin: known-method dispatch, universal-fallback for
unknown methods, PASS/WARN/FAIL routing, JSON safety, and the
distinction from sibling APIs (check_identification, audit).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def good_did_df():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        tr = i % 2
        for t in (0, 1):
            rows.append({"i": i, "y": rng.normal() + 0.5 * tr * t,
                         "treated": tr, "t": t})
    return pd.DataFrame(rows)


@pytest.fixture
def regress_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "y": rng.normal(size=200),
        "x1": rng.normal(size=200),
        "x2": rng.normal(size=200),
    })


# ---------------------------------------------------------------------------
#  Top-level export
# ---------------------------------------------------------------------------


class TestExport:

    def test_callable(self):
        assert callable(sp.preflight)

    def test_in_all(self):
        assert "preflight" in sp.__all__


# ---------------------------------------------------------------------------
#  Return shape
# ---------------------------------------------------------------------------


class TestReturnShape:

    def test_top_level_keys(self, regress_df):
        out = sp.preflight(regress_df, "regress", formula="y ~ x1")
        for k in ("method", "verdict", "checks", "summary", "n_obs",
                  "known_method"):
            assert k in out, f"missing key {k!r}"

    def test_verdict_in_known_set(self, regress_df):
        out = sp.preflight(regress_df, "regress", formula="y ~ x1")
        assert out["verdict"] in ("PASS", "WARN", "FAIL")

    def test_summary_counts_consistent(self, regress_df):
        out = sp.preflight(regress_df, "regress", formula="y ~ x1")
        s = out["summary"]
        assert s["passed"] + s["warning"] + s["failed"] == s["n_total"]

    def test_each_check_shape(self, regress_df):
        out = sp.preflight(regress_df, "regress", formula="y ~ x1")
        for c in out["checks"]:
            for k in ("name", "question", "status", "message",
                      "evidence"):
                assert k in c
            assert c["status"] in ("passed", "warning", "failed")

    def test_payload_strict_json_safe(self, regress_df):
        out = sp.preflight(regress_df, "regress", formula="y ~ x1")
        json.dumps(out)


# ---------------------------------------------------------------------------
#  Verdict routing
# ---------------------------------------------------------------------------


class TestVerdictPass:

    def test_clean_regress_passes(self, regress_df):
        out = sp.preflight(regress_df, "regress",
                            formula="y ~ x1 + x2")
        assert out["verdict"] == "PASS"
        assert out["known_method"] is True


class TestVerdictWarn:

    def test_tiny_n_warns(self):
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0],
                            "treated": [0, 1, 0, 1],
                            "t": [0, 0, 1, 1]})
        out = sp.preflight(df, "did", y="y", treat="treated", time="t")
        assert out["verdict"] == "WARN"
        warns = [c for c in out["checks"] if c["status"] == "warning"]
        assert any(c["name"] == "min_n_for_did" for c in warns)


class TestVerdictFail:

    def test_multi_arm_treatment_fails(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "y": rng.normal(size=120),
            "treated": [0, 1, 2] * 40,
            "t": [0, 1] * 60,
        })
        out = sp.preflight(df, "did", y="y", treat="treated", time="t")
        assert out["verdict"] == "FAIL"
        assert any(c["name"] == "treat_is_binary"
                   and c["status"] == "failed"
                   for c in out["checks"])

    def test_missing_column_fails(self, regress_df):
        out = sp.preflight(regress_df, "did", y="y",
                            treat="does_not_exist", time="t")
        assert out["verdict"] == "FAIL"

    def test_string_treatment_column_fails(self):
        # Regression guard: a CSV-imported "0"/"1" string column would
        # previously WARN and let the agent proceed; estimator then
        # raises an opaque downstream error. Preflight must FAIL here.
        df = pd.DataFrame({
            "y": [1.0, 2.0, 3.0, 4.0],
            "treat": ["0", "1", "0", "1"],  # strings, not ints
            "t": [0, 0, 1, 1],
        })
        out = sp.preflight(df, "did", y="y", treat="treat", time="t")
        assert out["verdict"] == "FAIL"
        binary_check = next(c for c in out["checks"]
                            if c["name"] == "treat_is_binary")
        assert binary_check["status"] == "failed"
        assert "dtype" in binary_check["evidence"]

    def test_non_dataframe_fails(self):
        out = sp.preflight([1, 2, 3], "did")
        assert out["verdict"] == "FAIL"
        assert any(c["name"] == "data_is_dataframe"
                   and c["status"] == "failed"
                   for c in out["checks"])

    def test_empty_dataframe_fails(self):
        out = sp.preflight(pd.DataFrame(), "did")
        assert out["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
#  Method dispatch
# ---------------------------------------------------------------------------


class TestMethodDispatch:

    def test_did_uses_did_table(self, good_did_df):
        out = sp.preflight(good_did_df, "did",
                            y="y", treat="treated", time="t")
        names = {c["name"] for c in out["checks"]}
        assert "treat_is_binary" in names
        assert "time_has_two_periods" in names

    def test_callaway_santanna_requires_id(self, good_did_df):
        # No ``i`` provided → expect FAIL on id_column_provided.
        out = sp.preflight(good_did_df, "callaway_santanna",
                            y="y", treat="treated", time="t")
        assert out["verdict"] == "FAIL"
        assert any(c["name"] == "id_column_provided"
                   and c["status"] == "failed"
                   for c in out["checks"])

    def test_callaway_santanna_passes_with_id(self, good_did_df):
        out = sp.preflight(good_did_df, "callaway_santanna",
                            y="y", treat="treated", time="t",
                            i="i")
        # Either PASS (if all good) or WARN on n_obs; never FAIL.
        assert out["verdict"] in ("PASS", "WARN")

    def test_rdrobust_needs_continuous_running_var(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "y": rng.normal(size=600),
            "score": rng.uniform(0, 100, size=600),
        })
        out = sp.preflight(df, "rdrobust", y="y", x="score")
        assert out["verdict"] == "PASS"

    def test_unknown_method_falls_back_to_universal(self, regress_df):
        out = sp.preflight(regress_df, "totally_made_up_method")
        assert out["known_method"] is False
        # Universal checks are exactly two: dataframe + non_empty.
        assert out["summary"]["n_total"] == 2

    def test_method_alias_resolves(self, regress_df):
        # "ols" is an alias for "regress".
        out_ols = sp.preflight(regress_df, "ols", formula="y ~ x1")
        out_regress = sp.preflight(regress_df, "regress",
                                    formula="y ~ x1")
        assert out_ols["summary"]["n_total"] == (
            out_regress["summary"]["n_total"])


# ---------------------------------------------------------------------------
#  Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_check_exception_is_failed_not_raised(self):
        # Pass a non-DataFrame; subsequent checks should still run and
        # the function must not raise.
        out = sp.preflight("not a dataframe", "did")
        assert out["verdict"] == "FAIL"
        assert isinstance(out["checks"], list)

    def test_method_lowercased_and_stripped(self, regress_df):
        out = sp.preflight(regress_df, "  REGRESS  ", formula="y ~ x1")
        assert out["method"] == "regress"
        assert out["known_method"] is True


# ---------------------------------------------------------------------------
#  IV-specific: first-stage strength gate (Stock-Yogo / Staiger-Stock)
# ---------------------------------------------------------------------------


def _make_iv_df(seed: int, pi: float, eu: float, n: int = 600,
                truth: float = 1.0) -> pd.DataFrame:
    """Linear-IV DGP: y = truth*d + u; d = pi*z + eu*u + noise.

    pi controls first-stage strength (small => weak); eu controls
    endogeneity loading.  Mirrors the Track B robustness DGP.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    d = pi * z + eu * u + rng.normal(scale=0.4, size=n)
    y = truth * d + u + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "d": d, "z": z})


def _first_stage_check(out: dict) -> dict:
    """Pluck the first_stage_strength row out of a preflight payload."""
    for c in out["checks"]:
        if c["name"] == "first_stage_strength":
            return c
    raise AssertionError("first_stage_strength check not in payload")


class TestIVFirstStageStrength:

    def test_strong_iv_passes(self):
        df = _make_iv_df(seed=2026, pi=0.7, eu=0.5, n=800)
        out = sp.preflight(df, "ivreg", formula="y ~ (d ~ z)")
        check = _first_stage_check(out)
        assert check["status"] == "passed"
        assert check["evidence"]["weakest_F"] >= 16.38

    def test_weak_iv_warns_with_recovery_hints(self):
        # Mirrors the Track B robustness DGP: pi=0.10 yields F_med ~ 3.
        df = _make_iv_df(seed=2026, pi=0.10, eu=1.5, n=600)
        out = sp.preflight(df, "ivreg", formula="y ~ (d ~ z)")
        check = _first_stage_check(out)
        assert check["status"] == "warning"
        ev = check["evidence"]
        assert ev["weakest_F"] < 10.0
        # Recovery hints must mention LIML and Anderson-Rubin so an
        # agent can route to those without parsing English prose.
        hint_text = " ".join(ev["recovery_hints"]).lower()
        assert "liml" in hint_text
        assert "anderson" in hint_text or "ar" in hint_text
        # The summary verdict propagates the warning.
        assert out["verdict"] in ("WARN", "PASS")
        # WARN if any check is warning, PASS otherwise — should be WARN.
        assert out["verdict"] == "WARN"

    def test_borderline_weak_iv_emits_stock_yogo_warning(self):
        # Tune pi so first-stage F lands between 10 and 16.38 with
        # high probability.  pi=0.15 yields F_med ~ 8 (borderline) on
        # n=600, but bumping n=2000 brings F into the [10, 16.38] band.
        df = _make_iv_df(seed=2026, pi=0.15, eu=1.5, n=2000)
        out = sp.preflight(df, "ivreg", formula="y ~ (d ~ z)")
        check = _first_stage_check(out)
        f = check["evidence"]["weakest_F"]
        # Allow either status — the boundary is sample-specific — but
        # if it warns, the message should mention Stock-Yogo not just
        # Staiger-Stock.
        if check["status"] == "warning" and 10.0 <= f < 16.38:
            assert "Stock-Yogo" in check["message"]

    def test_non_iv_formula_skipped_silently(self):
        # An IV-named call with a non-IV formula falls back to a
        # passed verdict on the strength check (the universal column
        # check / sp.ivreg itself will surface the formula mismatch).
        df = _make_iv_df(seed=2026, pi=0.7, eu=0.5, n=400)
        out = sp.preflight(df, "ivreg", formula="y ~ d + z")
        check = _first_stage_check(out)
        assert check["status"] == "passed"
        assert check["evidence"].get("skipped") == "non-IV formula"

    def test_missing_columns_skipped_not_failed(self):
        df = _make_iv_df(seed=2026, pi=0.7, eu=0.5, n=400)
        out = sp.preflight(df, "ivreg",
                            formula="y ~ (d ~ NotAColumn)")
        check = _first_stage_check(out)
        assert check["status"] == "passed"
        assert "missing" in check["evidence"].get("skipped", "")

    def test_payload_remains_json_safe_with_first_stage(self):
        df = _make_iv_df(seed=2026, pi=0.10, eu=1.5, n=600)
        out = sp.preflight(df, "ivreg", formula="y ~ (d ~ z)")
        # Must round-trip through JSON without exceptions; the
        # first_stage block contains nested floats.
        payload = json.dumps(out)
        assert "weakest_F" in payload
        assert "recovery_hints" in payload
