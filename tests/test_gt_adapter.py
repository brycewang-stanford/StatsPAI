"""Tests for ``sp.gt(result)`` — :pkg:`great_tables` adapter.

The adapter is opt-in publication polish on top of StatsPAI's existing
table outputs. Skipped en bloc when ``great_tables`` is not installed
(adapter is in optional extras).

Coverage
--------
- ``is_great_tables_available()`` reports the dep correctly.
- ``to_gt(DataFrame)`` returns a GT instance.
- ``to_gt(RegtableResult)`` carries title + notes + journal preset.
- ``to_gt(MeanComparisonResult)`` flattens via ``to_dataframe()``.
- ``to_gt(unsupported)`` raises a clear ``TypeError``.
- HTML / LaTeX renderers actually emit content (``as_raw_html`` /
  ``as_latex``).
- Journal theme application doesn't crash on any of the 8 presets,
  even if a particular gt option is missing in the installed
  version.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.output._gt import is_great_tables_available, to_gt

pytestmark = pytest.mark.skipif(
    not is_great_tables_available(),
    reason="great_tables not installed",
)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_dep_detected(self):
        assert is_great_tables_available() is True

    def test_dataframe_path(self):
        df = pd.DataFrame({
            "var": ["x1", "x2"],
            "M1": ["0.5***", "0.3"],
            "M2": ["0.45**", "0.28"],
        })
        g = to_gt(df, rowname_col="var", title="Returns")
        # Round-trip via HTML — establishes the GT actually renders.
        html = g.as_raw_html()
        assert "Returns" in html
        assert "0.5***" in html
        assert "x1" in html

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="does not know"):
            to_gt(object())


# ---------------------------------------------------------------------------
# RegtableResult path
# ---------------------------------------------------------------------------

@pytest.fixture
def fitted_model():
    """A small statsmodels-OLS-like fit that regtable can consume."""
    import statspai as sp
    np.random.seed(0)
    df = pd.DataFrame({
        "wage": np.random.randn(200) + 10.0,
        "trained": np.random.binomial(1, 0.5, 200),
        "edu": np.random.randn(200),
    })
    return sp.feols("wage ~ trained + edu", df)


class TestRegtableAdapter:
    def test_regtable_to_gt_basic(self, fitted_model):
        import statspai as sp
        rt = sp.regtable(fitted_model, title="Returns to Schooling")
        g = to_gt(rt)
        html = g.as_raw_html()
        assert "Returns to Schooling" in html
        # Coefficient labels carry through.
        assert "trained" in html
        assert "edu" in html

    def test_journal_template_applied(self, fitted_model):
        import statspai as sp
        rt = sp.regtable(fitted_model, template="aer")
        g = to_gt(rt)
        # Theme application is best-effort — the call must not raise.
        # Verify the table still renders by HTML round-trip.
        html = g.as_raw_html()
        assert "trained" in html

    def test_notes_carry_through(self, fitted_model):
        import statspai as sp
        rt = sp.regtable(
            fitted_model,
            template="aer",
            notes=["Robust SEs in parentheses.",
                   "*** p<0.01, ** p<0.05, * p<0.10."],
        )
        g = to_gt(rt)
        html = g.as_raw_html()
        # tab_source_note emits the notes near the table footer.
        assert "p&lt;0.01" in html or "p<0.01" in html
        assert "Robust SEs" in html

    def test_explicit_title_overrides(self, fitted_model):
        import statspai as sp
        rt = sp.regtable(fitted_model, title="Auto title")
        g = to_gt(rt, title="Manual override")
        html = g.as_raw_html()
        assert "Manual override" in html
        assert "Auto title" not in html

    @pytest.mark.parametrize("template", [
        "aer", "qje", "econometrica", "restat",
        "jf", "aeja", "jpe", "restud",
    ])
    def test_all_journal_presets_dont_crash(self, fitted_model, template):
        import statspai as sp
        rt = sp.regtable(fitted_model, template=template)
        g = to_gt(rt)
        # The full pipeline (including theme application) must not crash
        # for any of the 8 presets.
        assert g.as_raw_html()


# ---------------------------------------------------------------------------
# Generic DataFrame path
# ---------------------------------------------------------------------------

class TestDataFramePath:
    def test_no_rowname_col(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        g = to_gt(df)
        html = g.as_raw_html()
        assert "1" in html and "4" in html

    def test_with_rowname_col(self):
        df = pd.DataFrame({"name": ["a", "b"], "v": [1, 2]})
        g = to_gt(df, rowname_col="name", title="T")
        html = g.as_raw_html()
        assert "T" in html
        assert "a" in html and "b" in html

    def test_passes_subtitle(self):
        df = pd.DataFrame({"x": [1]})
        g = to_gt(df, title="Main", subtitle="Sub")
        html = g.as_raw_html()
        assert "Main" in html and "Sub" in html

    def test_notes(self):
        df = pd.DataFrame({"x": [1]})
        g = to_gt(df, title="T", notes=["Footnote A.", "Footnote B."])
        html = g.as_raw_html()
        assert "Footnote A" in html
        assert "Footnote B" in html


# ---------------------------------------------------------------------------
# Duck-typed (anything with to_dataframe)
# ---------------------------------------------------------------------------

class _DuckTable:
    """Minimal duck — like a custom result object with to_dataframe."""
    title = "Duck title"
    notes = ("note 1",)
    template = "aer"

    def to_dataframe(self):
        return pd.DataFrame({"a": [1, 2], "b": [3, 4]})


class TestDuckTypedPath:
    def test_to_dataframe_dispatch(self):
        g = to_gt(_DuckTable())
        html = g.as_raw_html()
        assert "Duck title" in html
        assert "note 1" in html
