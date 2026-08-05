"""Tests for journal-grade LaTeX: ``siunitx`` decimal alignment + threeparttable.

These are opt-in (``to_latex(siunitx=..., threeparttable=...)``); the default
LaTeX path must stay byte-identical (guarded by the existing regtable LaTeX
snapshot tests). Here we verify the opt-in structure follows the documented,
compiling ``siunitx`` v3 convention (coefficients in ``S`` columns,
``\\textsuperscript`` stars, braced SE / text cells) and that unsupported
feature combinations fail loudly.
"""

import re

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def table():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    df["y"] = 1.0 + 2.0 * df["x"] - 0.5 * df["z"] + rng.normal(size=n)
    df["y2"] = 0.5 + 1.5 * df["x"] + rng.normal(size=n)
    m1 = sp.regress("y ~ x + z", data=df)
    m2 = sp.regress("y2 ~ x", data=df)
    return sp.regtable(m1, m2, template="aer", model_labels=["A", "B"])


class TestDefaultUnchanged:
    def test_default_has_no_siunitx_columns(self, table):
        tex = table.to_latex()
        assert "S[table-format" not in tex
        assert "\\textsuperscript" not in tex
        assert tex.startswith("\\begin{table}")

    def test_default_notes_are_multicolumn_rows(self, table):
        tex = table.to_latex()
        assert "\\multicolumn" in tex
        assert "\\begin{tablenotes}" not in tex


class TestSiunitx:
    def test_s_columns_in_spec(self, table):
        tex = table.to_latex(siunitx=True)
        assert "S[table-format=" in tex
        assert "table-space-text-post={\\textsuperscript{***}}" in tex

    def test_table_format_has_sign_and_decimals(self, table):
        tex = table.to_latex(siunitx=True)
        tf = re.search(r"table-format=(-\d+\.\d+)", tex).group(1)
        # siunitx needs one fixed decimal count for the whole column, so the
        # adaptive default collapses to the finest precision any row asked
        # for. Assert the shape, not a specific count.
        assert tf.startswith("-")
        assert int(tf.split(".")[1]) >= 1

    def test_table_format_under_fixed_fmt(self):
        """An explicit printf fmt still drives table-format verbatim."""
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({"x": rng.normal(size=n)})
        df["y"] = 1.0 + 2.0 * df["x"] + rng.normal(size=n)
        m = sp.regress("y ~ x", data=df)
        tex = sp.regtable(m, fmt="%.3f").to_latex(siunitx=True)
        tf = re.search(r"table-format=(-\d+\.\d+)", tex).group(1)
        assert tf.endswith(".3")

    def test_siunitx_cells_have_no_thousands_separator(self):
        """Adaptive precision must not leak "1,234" into an S column.

        siunitx parses the cell as a number; a comma group separator would
        break alignment, so the siunitx path renders plainly.
        """
        rng = np.random.default_rng(7)
        n = 200
        df = pd.DataFrame({"x": rng.normal(size=n)})
        df["y"] = 5000.0 + 2500.0 * df["x"] + rng.normal(0, 500, n)
        m = sp.regress("y ~ x", data=df)
        tex = sp.regtable(m).to_latex(siunitx=True)
        body = [ln for ln in tex.splitlines() if "&" in ln and "textsuperscript" in ln]
        assert body, "no coefficient rows found"
        for line in body:
            assert "," not in line, f"separator leaked into S column: {line}"

    def test_decimals_track_fmt(self):
        rng = np.random.default_rng(1)
        n = 150
        df = pd.DataFrame({"x": rng.normal(size=n)})
        df["y"] = 1 + 2 * df["x"] + rng.normal(size=n)
        m = sp.regress("y ~ x", data=df)
        tex = sp.regtable(m, fmt="%.4f").to_latex(siunitx=True)
        assert re.search(r"table-format=-\d+\.4", tex)

    def test_starred_coef_uses_textsuperscript(self, table):
        tex = table.to_latex(siunitx=True)
        # a significant coefficient should render as <num>\textsuperscript{...}
        assert re.search(r"\d\\textsuperscript\{\*+\}", tex)

    def test_headers_wrapped_in_multicolumn(self, table):
        tex = table.to_latex(siunitx=True)
        assert "\\multicolumn{1}{c}{A}" in tex
        assert "\\multicolumn{1}{c}{B}" in tex

    def test_se_row_is_braced(self, table):
        tex = table.to_latex(siunitx=True)
        # SE cell like {(0.071)} so the S column treats it as text
        assert re.search(r"\{\([0-9.]+\)\}", tex)

    def test_preamble_comment(self, table):
        tex = table.to_latex(siunitx=True, siunitx_preamble=True)
        assert tex.startswith("% Preamble:")
        assert "\\usepackage{siunitx}" in tex.splitlines()[0]


class TestThreeparttable:
    def test_wrapper_and_tablenotes(self, table):
        tex = table.to_latex(threeparttable=True)
        assert "\\begin{threeparttable}" in tex
        assert "\\end{threeparttable}" in tex
        assert "\\begin{tablenotes}" in tex
        assert "\\end{tablenotes}" in tex
        assert "\\item " in tex

    def test_notes_not_multicolumn_rows(self, table):
        tex = table.to_latex(threeparttable=True)
        # the SE-in-parentheses note must be an \item, not a multicolumn row
        assert (
            "\\item Standard errors in parentheses" in tex
            or "\\item Robust standard errors" in tex
            or re.search(r"\\item .*in parentheses", tex)
        )

    def test_combined_with_siunitx(self, table):
        tex = table.to_latex(siunitx=True, threeparttable=True, siunitx_preamble=True)
        assert "\\begin{threeparttable}" in tex
        assert "S[table-format=" in tex
        assert tex.startswith("% Preamble:")
        assert "threeparttable" in tex.splitlines()[0]


class TestUnsupportedFeaturesRejected:
    @pytest.fixture
    def models(self):
        rng = np.random.default_rng(2)
        n = 300
        df = pd.DataFrame({"x": rng.normal(size=n)})
        df["y"] = 1 + 2 * df["x"] + rng.normal(size=n)
        df["bin"] = (df["y"] > df["y"].median()).astype(int)
        return df

    def test_eform_rejected(self, models):
        m = sp.logit("bin ~ x", data=models)
        with pytest.raises(NotImplementedError, match="eform"):
            sp.regtable(m, eform=True).to_latex(siunitx=True)

    def test_multi_se_rejected(self, models):
        m = sp.regress("y ~ x", data=models)
        se = {"Bootstrap SE": [{"x": 0.1, "Intercept": 0.1}]}
        with pytest.raises(NotImplementedError, match="multi_se"):
            sp.regtable(m, multi_se=se).to_latex(siunitx=True)

    def test_column_spanners_rejected(self, models):
        m1 = sp.regress("y ~ x", data=models)
        m2 = sp.regress("y ~ x", data=models)
        with pytest.raises(NotImplementedError, match="column_spanners"):
            sp.regtable(m1, m2, column_spanners=[("G", 2)]).to_latex(siunitx=True)


class TestEconometricResultsPassthrough:
    @pytest.fixture
    def ols(self):
        rng = np.random.default_rng(3)
        n = 200
        df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
        df["y"] = 1 + 2 * df["x"] - 0.5 * df["z"] + rng.normal(size=n)
        return sp.regress("y ~ x + z", data=df)

    def test_siunitx_threeparttable_via_result(self, ols):
        tex = ols.to_latex(siunitx=True, threeparttable=True)
        assert "S[table-format=" in tex
        assert "\\begin{threeparttable}" in tex

    def test_label_still_injected_with_siunitx(self, ols):
        tex = ols.to_latex(caption="Main", label="tab:m", siunitx=True)
        assert "\\caption{Main}" in tex
        assert "\\label{tab:m}" in tex

    def test_kwargs_still_forwarded(self, ols):
        tex = ols.to_latex(siunitx=True, coef_labels={"x": "Treatment"})
        assert "Treatment" in tex
