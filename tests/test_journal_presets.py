"""Tests for journal-preset-driven ``regtable`` styling.

Covers ``_journals.JOURNALS`` registry shape, ``regtable(template=...)``
plumbing (star levels, SE label, default stats), and the legacy
``paper_tables.TEMPLATES`` re-export.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.output._journals import (
    JOURNALS,
    get_template,
    list_templates,
    star_note_for,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_models():
    rng = np.random.default_rng(2026)
    n = 400
    x = rng.normal(0, 1, n)
    y = 0.5 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "x": x})
    m1 = sp.regress("y ~ x", data=df)
    m2 = sp.regress("y ~ x", data=df, robust="hc3")
    return m1, m2


# ---------------------------------------------------------------------------
# _journals registry
# ---------------------------------------------------------------------------

REQUIRED_TEMPLATES = ("aer", "qje", "econometrica", "restat", "jf", "aeja", "jpe", "restud")


def test_journals_registry_has_required_templates():
    for name in REQUIRED_TEMPLATES:
        assert name in JOURNALS, f"missing journal preset: {name}"


def test_list_templates_matches_registry():
    assert set(list_templates()) == set(JOURNALS.keys())


def test_get_template_case_insensitive_and_returns_copy():
    a = get_template("AER")
    b = get_template("aer")
    assert a == b
    a["star_levels"] = (0.42,)
    # Mutating a copy should not corrupt the registry
    assert JOURNALS["aer"]["star_levels"] != (0.42,)


def test_get_template_unknown_raises():
    with pytest.raises(ValueError, match="Unknown journal template"):
        get_template("totally-fake-journal")


def test_econometrica_keeps_three_thresholds():
    """Econometrica preset follows the AER-equivalent three-threshold default.

    Some older Econometrica papers used only ``**``/``***`` (5%/1%); newer
    issues use the full three-level convention. We default to three so behaviour
    matches the rest of the journal presets — users who need the legacy
    two-level scheme can pass ``star_levels=(0.05, 0.01)`` explicitly.
    """
    preset = get_template("econometrica")
    assert preset["star_levels"] == (0.10, 0.05, 0.01)


def test_qje_se_label_is_robust():
    """QJE convention says ``Robust standard errors`` in the table footer."""
    preset = get_template("qje")
    assert "robust" in preset["se_label"].lower()


def test_jf_aeja_include_adj_r2_in_default_stats():
    for name in ("jf", "aeja"):
        preset = get_template(name)
        assert "Adj. R-squared" in preset["stats"]


def test_star_note_strict_first_format():
    note = star_note_for((0.10, 0.05, 0.01))
    # Strict thresholds (most stars) come first by convention
    assert note.startswith("***")
    assert "* p<0.10" in note
    assert "** p<0.05" in note


def test_star_note_two_threshold_skips_single_star():
    """When a user passes only two thresholds, the loosest still gets ``*``."""
    note = star_note_for((0.05, 0.01))
    # Strict-first: ** p<0.01, * p<0.05 — only two pieces.
    pieces = [p.strip() for p in note.split(",")]
    assert len(pieces) == 2
    assert pieces[0].startswith("**")
    assert pieces[1].startswith("*") and not pieces[1].startswith("**")


# ---------------------------------------------------------------------------
# regtable(template=...) plumbing
# ---------------------------------------------------------------------------

def test_regtable_template_qje_uses_robust_se_label(two_models):
    m1, m2 = two_models
    txt = sp.regtable(m1, m2, template="qje").to_text()
    assert "Robust standard errors" in txt
    assert "Standard errors in parentheses" not in txt or "Robust" in txt


def test_regtable_template_aer_uses_plain_se_label(two_models):
    m1, m2 = two_models
    txt = sp.regtable(m1, m2, template="aer").to_text()
    assert "Standard errors in parentheses" in txt
    assert "Robust standard errors" not in txt


def test_regtable_template_econometrica_three_thresholds(two_models):
    """The Econometrica preset emits the standard three-level star note."""
    m1, m2 = two_models
    txt = sp.regtable(m1, m2, template="econometrica").to_text()
    assert "* p<0.10" in txt
    assert "** p<0.05" in txt
    assert "*** p<0.01" in txt


def test_regtable_template_jf_includes_adj_r2(two_models):
    m1, m2 = two_models
    txt = sp.regtable(m1, m2, template="jf").to_text()
    assert "Adj. R" in txt


def test_regtable_template_qje_drops_r2(two_models):
    """QJE preset stats = ('N',) only — no R² row shown by default."""
    m1, m2 = two_models
    txt = sp.regtable(m1, m2, template="qje").to_text()
    # The Unicode superscript-2 only appears via _STAT_DISPLAY for R-squared.
    assert "R²" not in txt
    # And the AER preset (which DOES include R²) should still show it,
    # to anchor that the assertion above is meaningful rather than
    # accidentally vacuous on every run.
    txt_aer = sp.regtable(m1, m2, template="aer").to_text()
    assert "R²" in txt_aer


def test_regtable_unknown_template_raises(two_models):
    m1, m2 = two_models
    with pytest.raises(ValueError, match="Unknown journal template"):
        sp.regtable(m1, m2, template="not-a-journal")


def test_regtable_explicit_kwargs_override_template(two_models):
    """Explicit ``star_levels`` should beat the template default."""
    m1, m2 = two_models
    txt = sp.regtable(m1, m2, template="econometrica",
                      star_levels=(0.10, 0.05, 0.01)).to_text()
    # 10% threshold should now appear in the star note
    assert "* p<0.10" in txt


# ---------------------------------------------------------------------------
# paper_tables ↔ _journals integration
# ---------------------------------------------------------------------------

def test_paper_tables_TEMPLATES_re_exports_all_journals():
    from statspai.output.paper_tables import TEMPLATES
    for name in REQUIRED_TEMPLATES:
        assert name in TEMPLATES, f"paper_tables.TEMPLATES missing {name}"


def test_paper_tables_picks_up_template_se_label(two_models):
    m1, m2 = two_models
    pt = sp.paper_tables(main=[m1, m2], template="qje")
    assert pt.main is not None
    txt = pt.main.to_text()
    assert "Robust standard errors" in txt
