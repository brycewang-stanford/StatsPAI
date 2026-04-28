"""
Round-3 publication extensions to ``regtable``.

Three additions complementing Rounds 1 & 2:

1. ``sp.margins_table(model)`` — adapter that returns a
   regtable-renderable result wrapping ``sp.margins`` output.
2. ``tests=`` — render hypothesis-test footer rows alongside main
   results with consistent formatting (p-values get stars, F-stats
   honor ``fmt``).
3. ``fixef_sizes=True`` — auto-emit "# Firm: 1,234 / # Year: 30"
   rows when ``model_info['n_fe_levels']`` is populated. Mirrors
   R ``fixest::etable``'s ``fixef.group + fixef_sizes`` family.
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
    y = 1.0 + 0.5 * x1 + 0.25 * x2 + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    m1 = sp.regress("y ~ x1", data=df)
    m2 = sp.regress("y ~ x1 + x2", data=df)
    return m1, m2, df


@pytest.fixture
def logit_models():
    rng = np.random.default_rng(7)
    n = 800
    x = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    p = 1.0 / (1.0 + np.exp(-(0.3 + 0.7 * x + 0.4 * z)))
    yb = (rng.uniform(size=n) < p).astype(int)
    df = pd.DataFrame({"yb": yb, "x": x, "z": z})
    return sp.logit("yb ~ x", data=df), sp.logit("yb ~ x + z", data=df), df


# ---------------------------------------------------------------------------
# 1. margins_table adapter
# ---------------------------------------------------------------------------

class TestMarginsTable:

    def test_margins_table_exists(self):
        assert hasattr(sp, "margins_table"), "sp.margins_table not registered"

    def test_margins_table_renders_text(self, logit_models):
        m1, m2, _ = logit_models
        mt1 = sp.margins_table(m1)
        mt2 = sp.margins_table(m2)
        out = sp.regtable(mt1, mt2).to_text()
        # Variable rows should be x and (in m2) z
        assert re.search(r"^\s*x\b", out, re.M), \
            f"x row missing in:\n{out}"

    def test_margins_table_numbers_match_margins_df(self, logit_models):
        m, _, _ = logit_models
        mfx_df = sp.margins(m)
        mt = sp.margins_table(m)
        # The mt result's params series should match mfx_df['dy/dx']
        # for variable 'x'
        x_dydx = float(
            mfx_df.loc[mfx_df["variable"] == "x", "dy/dx"].iloc[0]
        )
        # Adapter exposes .params (pandas Series indexed by variable name)
        assert "x" in mt.params.index
        assert abs(float(mt.params["x"]) - x_dydx) < 1e-10

    def test_margins_table_se_match(self, logit_models):
        m, _, _ = logit_models
        mfx_df = sp.margins(m)
        mt = sp.margins_table(m)
        x_se = float(
            mfx_df.loc[mfx_df["variable"] == "x", "se"].iloc[0]
        )
        assert abs(float(mt.std_errors["x"]) - x_se) < 1e-10

    def test_margins_table_renders_latex(self, logit_models):
        m, _, _ = logit_models
        tex = sp.regtable(sp.margins_table(m)).to_latex()
        assert "\\begin{tabular}" in tex
        assert "\\end{tabular}" in tex

    def test_margins_table_method_kwarg_propagates(self, logit_models):
        m, _, _ = logit_models
        # method='ame' default vs method='mer' — both should yield a table
        mt = sp.margins_table(m, method="ame")
        out = sp.regtable(mt).to_text()
        assert "x" in out


# ---------------------------------------------------------------------------
# 2. tests= footer rows
# ---------------------------------------------------------------------------

class TestTestsFooter:

    def test_tests_dict_renders_as_rows(self, ols_models):
        m1, m2, _ = ols_models
        out = sp.regtable(
            m1, m2,
            tests={
                "F-test x1=0": [(12.34, 0.0001), (8.91, 0.003)],
            },
        ).to_text()
        assert "F-test x1=0" in out
        # Statistic should render with 3 decimals
        assert "12.340" in out
        # p-value should render with stars
        assert "***" in out

    def test_tests_pvalue_alone_gets_stars(self, ols_models):
        m1, m2, _ = ols_models
        out = sp.regtable(
            m1, m2,
            tests={"Hansen J p-value": [0.04, 0.62]},
        ).to_text()
        assert "Hansen J p-value" in out
        # 0.04 should get one star
        assert re.search(r"0\.040\*\*", out) or re.search(r"0\.040\*", out)

    def test_tests_in_latex_and_html(self, ols_models):
        m1, m2, _ = ols_models
        for fmt_method in ("to_latex", "to_html"):
            r = sp.regtable(m1, m2, tests={"Wald F": [10.5, 22.1]})
            out = getattr(r, fmt_method)()
            assert "Wald F" in out
            assert "10.500" in out

    def test_tests_validates_per_model_length(self, ols_models):
        m1, m2, _ = ols_models
        with pytest.raises(ValueError):
            sp.regtable(m1, m2, tests={"Bad": [1.0]})


# ---------------------------------------------------------------------------
# 3. fixef_sizes
# ---------------------------------------------------------------------------

class TestFixefSizes:

    def _result_with_n_fe_levels(self, base_result, levels_dict):
        """Wrap a fitted result so model_info exposes n_fe_levels.

        We rely on the public ``EconometricResults`` shape — the adapter
        mutates ``model_info`` in place to inject the level dict. If the
        underlying class is immutable we fall back to a duck-typed
        wrapper.
        """
        try:
            mi = base_result.model_info
            mi["n_fe_levels"] = levels_dict
            mi["fixed_effects"] = "+".join(levels_dict.keys())
            return base_result
        except (AttributeError, TypeError):
            class _Wrapper:
                params = base_result.params
                std_errors = base_result.std_errors
                pvalues = getattr(base_result, "pvalues", None)
                tvalues = getattr(base_result, "tvalues", None)
                conf_int_lower = getattr(base_result, "conf_int_lower", None)
                conf_int_upper = getattr(base_result, "conf_int_upper", None)
                diagnostics = getattr(base_result, "diagnostics", {})
                data_info = getattr(base_result, "data_info", {})
                model_info = {
                    **getattr(base_result, "model_info", {}),
                    "fixed_effects": "+".join(levels_dict.keys()),
                    "n_fe_levels": levels_dict,
                }
            return _Wrapper()

    def test_fixef_sizes_emits_count_rows(self, ols_models):
        m1, _, _ = ols_models
        r = self._result_with_n_fe_levels(m1, {"firm": 1234, "year": 30})
        out = sp.regtable(r, fixef_sizes=True).to_text()
        # Both level-count rows should appear
        assert "Firm" in out
        assert "Year" in out
        assert "1,234" in out  # thousands separator
        assert "30" in out

    def test_fixef_sizes_off_by_default(self, ols_models):
        m1, _, _ = ols_models
        r = self._result_with_n_fe_levels(m1, {"firm": 1234})
        # Without fixef_sizes=True, count rows should NOT appear
        out = sp.regtable(r).to_text()
        assert "1,234" not in out

    def test_fixef_sizes_no_op_when_not_populated(self, ols_models):
        m1, _, _ = ols_models
        # Plain OLS without n_fe_levels — fixef_sizes=True should be a
        # silent no-op (no extra rows, no exception)
        out = sp.regtable(m1, fixef_sizes=True).to_text()
        assert "x1" in out  # normal rendering survives

    def test_fixef_sizes_in_latex(self, ols_models):
        m1, _, _ = ols_models
        r = self._result_with_n_fe_levels(m1, {"state": 50})
        tex = sp.regtable(r, fixef_sizes=True).to_latex()
        assert "State" in tex
        assert "50" in tex
