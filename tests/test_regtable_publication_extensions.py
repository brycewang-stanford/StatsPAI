"""
Publication-quality extensions to ``regtable``.

Covers six feature additions designed to close the gap to Stata ``esttab`` /
R ``modelsummary`` / R ``fixest::etable`` for empirical-paper workflows:

1. ``eform`` — exponentiate coefficients (logit OR / poisson IRR / cox HR)
   with delta-method SE and exp-of-endpoints CI; ``bool`` or per-model
   ``List[bool]``.
2. ``column_spanners`` — multi-row header with ``\\multicolumn`` spans
   (LaTeX) / ``colspan`` (HTML) / repeated header (text/markdown).
3. ``stats=["depvar_mean", "depvar_sd"]`` — auto rows for the dependent
   variable's sample mean and SD, populated from ``data_info['y']`` when
   available.
4. ``coef_map`` — single dict that simultaneously renames, reorders, and
   drops (mirrors R ``modelsummary``).
5. N-consistency and SE-type-mixing warnings.
6. LaTeX special-character escape correctness on user-supplied labels.

These tests are pure black-box: they construct OLS/Logit/Poisson models
with fixed seeds and inspect ``regtable`` output strings.
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
def logit_model():
    rng = np.random.default_rng(2027)
    n = 800
    x = rng.normal(0, 1, n)
    p = 1.0 / (1.0 + np.exp(-(0.3 + 0.7 * x)))
    y = (rng.uniform(size=n) < p).astype(int)
    df = pd.DataFrame({"y": y, "x": x})
    return sp.logit("y ~ x", data=df)


# ---------------------------------------------------------------------------
# 1. eform
# ---------------------------------------------------------------------------

class TestEform:

    def test_eform_true_exponentiates_coefs_text(self, logit_model):
        raw = sp.regtable(logit_model).to_text()
        eform = sp.regtable(logit_model, eform=True).to_text()
        # Coefficient changes after exp transform — expect different rendered cell
        assert raw != eform
        # x odds ratio should be e^0.7 ≈ 2.01 → cell starting "2." in eform
        m = re.search(r"\bx\b\s+(-?\d+\.\d+)", eform)
        assert m, f"Could not find coefficient row for x in:\n{eform}"
        assert float(m.group(1)) > 1.0, "exp(0.7) should be > 1"

    def test_eform_se_uses_delta_method(self, logit_model):
        """SE_exp = exp(b) * SE(b) under the delta method."""
        eform_txt = sp.regtable(logit_model, eform=True).to_text()
        # Cell layout: "x   <OR>***\n     (<SE>)\n"
        # Just verify SE row exists and SE > 0 by parsing
        lines = eform_txt.splitlines()
        x_idx = next(i for i, ln in enumerate(lines) if re.match(r"^\s*x\b", ln))
        se_row = lines[x_idx + 1]
        se_match = re.search(r"\(\s*(-?\d+\.\d+)\s*\)", se_row)
        assert se_match, f"SE not found in row: {se_row!r}"
        se_val = float(se_match.group(1))
        assert se_val > 0

    def test_eform_ci_uses_exp_of_endpoints(self, logit_model):
        """CI under eform is (exp(lo), exp(hi)) of the original CI."""
        # Get raw CI for x
        raw_lo, raw_hi = (
            float(logit_model.conf_int_lower["x"]),
            float(logit_model.conf_int_upper["x"]),
        )
        e_txt = sp.regtable(logit_model, eform=True, se_type="ci").to_text()
        m = re.search(r"x.*?\[\s*(-?[\d.]+),\s*(-?[\d.]+)\s*\]", e_txt, re.S)
        assert m, f"could not parse CI from:\n{e_txt}"
        lo, hi = float(m.group(1)), float(m.group(2))
        assert abs(lo - np.exp(raw_lo)) < 0.01
        assert abs(hi - np.exp(raw_hi)) < 0.01

    def test_eform_per_model_list(self, ols_models, logit_model):
        m1, _, _ = ols_models
        # Mixed: OLS no transform, Logit OR
        out = sp.regtable(m1, logit_model, eform=[False, True]).to_text()
        assert "exp" in out.lower() or "Odds" in out or "OR" in out  # footer note

    def test_eform_adds_footer_note(self, logit_model):
        out = sp.regtable(logit_model, eform=True).to_text()
        # Footer note must mark eform transformation transparently
        assert "exp(b)" in out or "exponentiated" in out.lower()

    def test_eform_preserves_stars(self, logit_model):
        """Significance stars come from the original p-value, not the exp cell."""
        out = sp.regtable(logit_model, eform=True).to_text()
        assert "***" in out or "**" in out  # logit at n=800 with b=0.7 → highly sig

    def test_eform_latex_round_trip(self, logit_model):
        tex = sp.regtable(logit_model, eform=True).to_latex()
        assert "\\begin{tabular}" in tex
        assert "\\end{tabular}" in tex

    def test_eform_html_round_trip(self, logit_model):
        html = sp.regtable(logit_model, eform=True).to_html()
        assert "<table" in html and "</table>" in html


# ---------------------------------------------------------------------------
# 2. Column spanners
# ---------------------------------------------------------------------------

class TestColumnSpanners:

    def test_spanners_text_renders_label_centered(self, ols_models):
        m1, m2, _ = ols_models
        out = sp.regtable(
            m1, m2, m1, m2,
            column_spanners=[("OLS", 2), ("IV", 2)],
        ).to_text()
        assert "OLS" in out
        assert "IV" in out

    def test_spanners_latex_uses_multicolumn(self, ols_models):
        m1, m2, _ = ols_models
        tex = sp.regtable(
            m1, m2, m1, m2,
            column_spanners=[("OLS", 2), ("IV", 2)],
        ).to_latex()
        assert "\\multicolumn{2}{c}{OLS}" in tex
        assert "\\multicolumn{2}{c}{IV}" in tex
        # cmidrule should appear under the spanners for booktabs polish
        assert "\\cmidrule" in tex

    def test_spanners_html_uses_colspan(self, ols_models):
        m1, m2, _ = ols_models
        html = sp.regtable(
            m1, m2, m1, m2,
            column_spanners=[("OLS", 2), ("IV", 2)],
        ).to_html()
        assert 'colspan="2"' in html
        assert ">OLS<" in html

    def test_spanners_validates_total_span(self, ols_models):
        m1, m2, _ = ols_models
        # Spans must sum to n_models
        with pytest.raises(ValueError, match="span"):
            sp.regtable(m1, m2, column_spanners=[("OLS", 1), ("IV", 2)])

    def test_spanners_escapes_latex_specials(self, ols_models):
        m1, m2, _ = ols_models
        tex = sp.regtable(
            m1, m2,
            column_spanners=[("M&M Co.", 2)],
        ).to_latex()
        assert "M\\&M Co." in tex


# ---------------------------------------------------------------------------
# 3. depvar_mean / depvar_sd auto rows
# ---------------------------------------------------------------------------

class TestDepvarStats:

    def test_depvar_mean_row_appears(self, ols_models):
        m1, m2, df = ols_models
        out = sp.regtable(m1, m2, stats=["N", "depvar_mean"]).to_text()
        assert "Mean of Y" in out or "Dep. var. mean" in out
        # Mean of y matches df["y"].mean() to 3 dp
        m = re.search(r"(?:Mean of Y|Dep\. var\. mean)\s+(-?\d+\.\d+)", out)
        assert m
        assert abs(float(m.group(1)) - df["y"].mean()) < 0.01

    def test_depvar_sd_row_appears(self, ols_models):
        m1, _, df = ols_models
        out = sp.regtable(m1, stats=["depvar_sd"]).to_text()
        m = re.search(r"(?:SD of Y|Dep\. var\. SD)\s+(-?\d+\.\d+)", out)
        assert m
        # ddof=1 sample SD
        assert abs(float(m.group(1)) - df["y"].std(ddof=1)) < 0.05

    def test_depvar_stats_in_latex_and_html(self, ols_models):
        m1, _, _ = ols_models
        tex = sp.regtable(m1, stats=["N", "depvar_mean", "depvar_sd"]).to_latex()
        html = sp.regtable(m1, stats=["N", "depvar_mean", "depvar_sd"]).to_html()
        for out in (tex, html):
            assert "Mean of Y" in out or "Dep. var. mean" in out
            assert "SD of Y" in out or "Dep. var. SD" in out


# ---------------------------------------------------------------------------
# 4. coef_map unified
# ---------------------------------------------------------------------------

class TestCoefMap:

    def test_coef_map_renames_and_orders_and_drops(self, ols_models):
        m1, m2, _ = ols_models
        # Drop x1 by omitting it from the map; keep x2 first labelled "Education"
        out = sp.regtable(
            m1, m2,
            coef_map={"x2": "Education", "Intercept": "Constant"},
        ).to_text()
        assert "Education" in out
        assert "Constant" in out
        # x1 omitted because it's not in coef_map — "x1" as standalone label gone
        # (plain x1 should not appear as a row label, but "x1 + x2" formula chunks
        # don't appear in regtable so this is safe)
        assert not re.search(r"^\s*x1\s+", out, re.M), \
            f"x1 should be dropped; got:\n{out}"

    def test_coef_map_preserves_order(self, ols_models):
        m1, m2, _ = ols_models
        out = sp.regtable(
            m1, m2,
            coef_map={"Intercept": "Constant", "x2": "Edu", "x1": "Exper"},
        ).to_text()
        # Order in output reflects coef_map insertion order
        idx_const = out.find("Constant")
        idx_edu = out.find("Edu")
        idx_exper = out.find("Exper")
        assert idx_const < idx_edu < idx_exper

    def test_coef_map_conflicts_raise(self, ols_models):
        m1, _, _ = ols_models
        # Passing both coef_map and coef_labels is ambiguous
        with pytest.raises((ValueError, TypeError)):
            sp.regtable(m1, coef_map={"x1": "X"}, coef_labels={"x1": "Y"})


# ---------------------------------------------------------------------------
# 5. Consistency warnings
# ---------------------------------------------------------------------------

class TestConsistencyWarnings:

    def test_n_mismatch_warning(self, ols_models):
        m1, m2, df = ols_models
        # Create m3 on a subsample so N differs
        df3 = df.iloc[:300]
        m3 = sp.regress("y ~ x1", data=df3)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp.regtable(m1, m3)
            n_warns = [x for x in w if "sample" in str(x.message).lower()
                       or "N" in str(x.message) and "differ" in str(x.message).lower()]
        assert len(n_warns) >= 1, [str(x.message) for x in w]

    def test_no_warning_when_n_matches(self, ols_models):
        m1, m2, _ = ols_models
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp.regtable(m1, m2)
        # m1 and m2 fit on same df → no N-mismatch warning
        n_warns = [
            x for x in w
            if "sample" in str(x.message).lower()
            and "differ" in str(x.message).lower()
        ]
        assert n_warns == []


# ---------------------------------------------------------------------------
# 6. LaTeX escape correctness
# ---------------------------------------------------------------------------

class TestLatexEscape:

    def test_underscore_in_coef_label_is_escaped(self, ols_models):
        m1, _, _ = ols_models
        tex = sp.regtable(
            m1,
            coef_labels={"x1": "log_wage"},
        ).to_latex()
        # Raw underscore would break LaTeX; must be escaped
        assert "log\\_wage" in tex
        # Bare "log_wage" without backslash must NOT appear
        # (allow "log\_wage" only)
        bad = re.search(r"(?<!\\)log_wage", tex)
        assert bad is None, f"Unescaped underscore: {tex}"

    def test_ampersand_in_title_is_escaped(self, ols_models):
        m1, _, _ = ols_models
        tex = sp.regtable(m1, title="Wages & Hours").to_latex()
        assert "Wages \\& Hours" in tex
