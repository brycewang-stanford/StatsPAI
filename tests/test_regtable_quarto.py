"""
Tests for ``regtable(..., quarto_label=...)`` Quarto cross-reference output.

Quarto manuscripts reference tables via ``@tbl-<label>``. ``regtable`` emits
the canonical ``: <caption> {#tbl-<label>}`` block that wires up the link.
"""

import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp


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
    return m1, m2


def test_to_quarto_emits_caption_block(ols_models):
    m1, m2 = ols_models
    r = sp.regtable(m1, m2, quarto_label="main", quarto_caption="Wage")
    out = r.to_quarto()
    assert ": Wage {#tbl-main}" in out


def test_label_auto_prepends_tbl_prefix(ols_models):
    m1, _ = ols_models
    r = sp.regtable(m1, quarto_label="wage", quarto_caption="W")
    assert "{#tbl-wage}" in r.to_quarto()


def test_label_already_prefixed_not_doubled(ols_models):
    m1, _ = ols_models
    r = sp.regtable(m1, quarto_label="tbl-already", quarto_caption="A")
    out = r.to_quarto()
    assert "{#tbl-already}" in out
    assert "{#tbl-tbl-" not in out


def test_caption_falls_back_to_title(ols_models):
    m1, _ = ols_models
    r = sp.regtable(m1, quarto_label="fb", title="My Title")
    assert ": My Title {#tbl-fb}" in r.to_quarto()


def test_missing_label_raises(ols_models):
    m1, _ = ols_models
    r = sp.regtable(m1, quarto_caption="C")
    with pytest.raises(ValueError, match="quarto_label"):
        r.to_quarto()


def test_missing_caption_warns_and_uses_default(ols_models):
    m1, _ = ols_models
    r = sp.regtable(m1, quarto_label="x")
    with pytest.warns(UserWarning, match="quarto_caption"):
        out = r.to_quarto()
    assert ": Regression results {#tbl-x}" in out


def test_output_quarto_dispatches_via_str(ols_models):
    m1, _ = ols_models
    r = sp.regtable(
        m1, quarto_label="main", quarto_caption="C", output="quarto"
    )
    assert "{#tbl-main}" in str(r)


def test_to_markdown_quarto_kwarg_equivalent(ols_models):
    m1, _ = ols_models
    r = sp.regtable(m1, quarto_label="m", quarto_caption="C")
    assert r.to_markdown(quarto=True) == r.to_quarto()


def test_save_qmd_extension_writes_quarto(ols_models):
    m1, _ = ols_models
    r = sp.regtable(m1, quarto_label="save", quarto_caption="C")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "table.qmd"
        r.save(str(path))
        content = path.read_text(encoding="utf-8")
    assert "{#tbl-save}" in content
    assert ": C {" in content


def test_quarto_does_not_duplicate_title_in_body(ols_models):
    m1, _ = ols_models
    r = sp.regtable(m1, quarto_label="d", title="My Heading")
    out = r.to_quarto()
    # Title is rendered as caption, not as a bold heading line.
    assert "**My Heading**" not in out


def test_to_markdown_default_unchanged(ols_models):
    m1, _ = ols_models
    r = sp.regtable(m1, title="Plain")
    md = r.to_markdown()
    assert "**Plain**" in md
    assert "{#tbl-" not in md


def test_quarto_output_contains_table_body(ols_models):
    m1, m2 = ols_models
    r = sp.regtable(m1, m2, quarto_label="main", quarto_caption="C")
    out = r.to_quarto()
    # Expect both column headers and at least one coefficient row.
    assert "(1)" in out and "(2)" in out
    assert "x1" in out


def test_invalid_output_quarto_typo_still_validated(ols_models):
    m1, _ = ols_models
    with pytest.raises(ValueError, match="invalid"):
        sp.regtable(m1, quarto_label="x", output="quartoo")
