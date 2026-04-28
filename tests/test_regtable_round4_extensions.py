"""
Round-4 publication extensions to ``regtable``.

Three additions on top of Rounds 1-3:

1. ``sp.event_study_table(result)`` — adapter that reads
   ``model_info['event_study']`` (DiD CausalResult) or extracts
   event-time coefs via regex from raw ``params.index``, returns a
   regtable-renderable result with rows ordered by relative time.

2. ``vcov="HC0" | "HC1" | "HC2" | "HC3" | "robust"`` — for OLS,
   recompute SE / t / p / CI at ``regtable`` print time using the
   stored X + residuals. Stata's ``robust`` ≈ HC1; we accept both
   spellings. Non-OLS results raise an informative error.

3. ``transpose=True`` — single-panel pivot. Rows become models,
   columns become variables. Multi-panel / multi_se tables raise
   to keep semantics tight.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ols_models():
    rng = np.random.default_rng(2026)
    n = 600
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    # Heteroskedastic so HC SEs differ from OLS SEs
    y = 1.0 + 0.5 * x1 + 0.25 * x2 + (1 + np.abs(x1)) * rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    m1 = sp.regress("y ~ x1", data=df)
    m2 = sp.regress("y ~ x1 + x2", data=df)
    return m1, m2, df


@pytest.fixture
def event_study_result():
    rng = np.random.default_rng(0)
    n_units = 80
    n_periods = 10
    rows = []
    for u in range(n_units):
        treat_time = 5 if u < 40 else np.nan
        for t in range(n_periods):
            post = (
                (t >= treat_time)
                if not np.isnan(treat_time)
                else False
            )
            y = (
                (1.0 if post else 0.0) * 0.5
                + rng.normal(0, 0.5)
                + u * 0.02
                + t * 0.1
            )
            rows.append({
                "unit": u, "time": t, "treat_time": treat_time, "y": y,
            })
    df = pd.DataFrame(rows)
    return sp.event_study(
        df, y="y", treat_time="treat_time", time="time",
        unit="unit", window=(-3, 3),
    )


# ---------------------------------------------------------------------------
# 1. event_study_table
# ---------------------------------------------------------------------------

class TestEventStudyTable:

    def test_event_study_table_exists(self):
        assert hasattr(sp, "event_study_table"), \
            "sp.event_study_table not registered"

    def test_extracts_relative_time_rows(self, event_study_result):
        et = sp.event_study_table(event_study_result)
        # Rows for relative times -3..-1, 0..3 (ref_period -1 has zero estimate)
        labels = list(et.params.index)
        # Should contain entries spanning negative and non-negative times
        assert any("-3" in s or "−3" in s for s in labels)
        assert any("0" in s for s in labels)
        assert any("3" in s and "-" not in s and "−" not in s for s in labels)

    def test_event_study_table_renders(self, event_study_result):
        et = sp.event_study_table(event_study_result)
        out = sp.regtable(et, title="Event study").to_text()
        assert "Event study" in out
        # Estimate at t=0 should be sizeable (≈0.66 in fixture)
        assert "0.6" in out or "0.7" in out

    def test_event_study_table_orders_by_relative_time(self, event_study_result):
        et = sp.event_study_table(event_study_result)
        # The display order should be -3, -2, -1, 0, 1, 2, 3
        labels = list(et.params.index)
        # Extract integer time from each label (handles "t=-3", "t_-3", etc.)
        times = []
        for lbl in labels:
            m = re.search(r"(-?\d+)", lbl)
            assert m, f"Could not parse time from label {lbl!r}"
            times.append(int(m.group(1)))
        assert times == sorted(times), \
            f"Labels not time-sorted: {labels}"

    def test_event_study_table_se_match(self, event_study_result):
        et = sp.event_study_table(event_study_result)
        es_df = event_study_result.model_info["event_study"]
        # Pick any non-reference row and verify SE matches
        for idx, row in es_df.iterrows():
            if abs(row["se"]) > 1e-10:
                # Find matching label in et.params
                for lbl in et.params.index:
                    m = re.search(r"(-?\d+)", lbl)
                    if m and int(m.group(1)) == int(row["relative_time"]):
                        assert abs(et.std_errors[lbl] - row["se"]) < 1e-10
                        return
        pytest.fail("No non-zero SE row found")

    def test_event_study_regex_path(self, ols_models):
        """Regex extraction from raw param names (no model_info path)."""
        m1, _, _ = ols_models
        # Synthesise a duck-typed result with event-style names — keeps
        # the test independent of which attributes EconometricResults
        # exposes as Series vs ndarray internally.
        idx = ["Intercept", "tau_-1"]
        b = pd.Series([float(m1.params["Intercept"]), float(m1.params["x1"])], index=idx)
        se = pd.Series(
            [float(m1.std_errors["Intercept"]), float(m1.std_errors["x1"])], index=idx,
        )
        zero = pd.Series(np.zeros(2), index=idx)

        class _DuckResult:
            params = b
            std_errors = se
            tvalues = b / se
            pvalues = pd.Series([0.5, 0.001], index=idx)
            conf_int_lower = zero
            conf_int_upper = zero
            diagnostics: dict = {}
            data_info: dict = {}
            model_info: dict = {}

        et = sp.event_study_table(_DuckResult(), regex=r"^tau_(-?\d+)$")
        # Should have one event-time row (only "tau_-1" matches)
        assert len(et.params) == 1


# ---------------------------------------------------------------------------
# 2. vcov= print-time recompute (HC0/HC1/HC2/HC3)
# ---------------------------------------------------------------------------

class TestVcovRecompute:

    def test_vcov_HC1_changes_se_under_heteroskedasticity(self, ols_models):
        m1, _, _ = ols_models
        # Default OLS SE
        out_default = sp.regtable(m1).to_text()
        # HC1 SE
        out_HC1 = sp.regtable(m1, vcov="HC1").to_text()
        # Under heteroskedasticity, HC1 SE differs from OLS SE
        assert out_default != out_HC1

    def test_vcov_robust_alias_matches_HC1(self, ols_models):
        m1, _, _ = ols_models
        out_robust = sp.regtable(m1, vcov="robust").to_text()
        out_HC1 = sp.regtable(m1, vcov="HC1").to_text()
        # Stata's ``robust`` ≡ HC1
        assert out_robust == out_HC1

    def test_vcov_HC3_se_typically_largest(self, ols_models):
        m1, _, _ = ols_models
        # Recompute HC0/HC1/HC2/HC3 SEs and confirm HC3 ≥ HC2 ≥ HC1 ≥ HC0
        # for at least one coefficient (typical under heteroskedasticity).
        ses = {}
        for v in ("HC0", "HC1", "HC2", "HC3"):
            txt = sp.regtable(m1, vcov=v).to_text()
            # Parse SE for x1 (second coef row, line with "(...)" right
            # below the x1 value row)
            lines = txt.splitlines()
            for i, ln in enumerate(lines):
                if re.match(r"^\s*x1\b", ln):
                    se_line = lines[i + 1]
                    m = re.search(r"\(\s*(-?\d+\.\d+)\s*\)", se_line)
                    if m:
                        ses[v] = float(m.group(1))
                    break
        assert ses["HC3"] >= ses["HC2"], f"HC3<HC2: {ses}"
        assert ses["HC2"] >= ses["HC0"], f"HC2<HC0: {ses}"

    def test_vcov_invalid_raises(self, ols_models):
        m1, _, _ = ols_models
        with pytest.raises((ValueError, NotImplementedError)):
            sp.regtable(m1, vcov="frobnicated")

    def test_vcov_t_and_p_recomputed(self, ols_models):
        m1, _, _ = ols_models
        # Render with HC3, parse the t-value to ensure it's recomputed
        # rather than left unchanged
        out_default = sp.regtable(m1, se_type="t").to_text()
        out_HC3 = sp.regtable(m1, se_type="t", vcov="HC3").to_text()
        assert out_default != out_HC3


# ---------------------------------------------------------------------------
# 3. transpose=True
# ---------------------------------------------------------------------------

class TestTranspose:

    def test_transpose_text_swaps_axes(self, ols_models):
        m1, m2, _ = ols_models
        # Default: rows = vars (Intercept, x1, x2), cols = (1), (2)
        # Transposed: rows = (1), (2), cols = vars
        normal = sp.regtable(m1, m2).to_text()
        flipped = sp.regtable(m1, m2, transpose=True).to_text()
        # Normal layout has "x1" as a row label
        assert re.search(r"^\s*x1\b", normal, re.M)
        # Flipped layout has model labels as rows ("(1)", "(2)") and
        # variable names appear in the column headers area
        assert re.search(r"^\s*\(1\)", flipped, re.M)

    def test_transpose_html_round_trip(self, ols_models):
        m1, m2, _ = ols_models
        html = sp.regtable(m1, m2, transpose=True).to_html()
        assert "<table" in html and "</table>" in html

    def test_transpose_rejects_multi_panel(self, ols_models):
        m1, m2, _ = ols_models
        with pytest.raises((ValueError, NotImplementedError)):
            sp.regtable([m1, m2], [m1, m2], transpose=True)

    def test_transpose_rejects_multi_se(self, ols_models):
        m1, m2, _ = ols_models
        # multi_se interacts with the cell layout; reject for now to
        # keep semantics tight.
        boot = pd.Series({"x1": 0.05}, dtype=float)
        with pytest.raises((ValueError, NotImplementedError)):
            sp.regtable(
                m1, m2,
                multi_se={"Boot SE": [boot, boot]},
                transpose=True,
            )
