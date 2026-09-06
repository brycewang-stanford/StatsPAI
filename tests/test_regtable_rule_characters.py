"""Plain-text regression tables must survive a legacy console code page.

``print(sp.regtable(m))`` built its rules from U+2501/U+2500, which no
8-bit code page contains: on a Windows console running cp1252 the call
raised ``UnicodeEncodeError`` before printing a single row. These tests
pin the three ``rules=`` modes and the auto-detection that makes the
default safe without changing what a UTF-8 terminal sees.
"""

from __future__ import annotations

import io
import sys

import pytest

import statspai as sp


@pytest.fixture(scope="module")
def models():
    card = sp.datasets.card_1995()
    return (
        sp.regress("lwage ~ educ + exper", data=card),
        sp.iv("lwage ~ exper + black + south + (educ ~ nearc4)", data=card),
    )


def _as_cp1252(text: str) -> bytes:
    return text.encode("cp1252")


class TestRegtableRuleCharacters:
    def test_ascii_mode_output_is_pure_ascii(self, models):
        text = str(sp.regtable(*models, rules="ascii"))
        assert text.isascii()
        text.encode("ascii")
        _as_cp1252(text)
        assert "=" * 20 in text
        # The superscript of the R-squared label folds too, and folds to a
        # same-width spelling so the fixed-width columns stay aligned.
        assert "R2" in text and "R²" not in text

    def test_ascii_mode_preserves_the_numbers(self, models):
        unicode_text = str(sp.regtable(*models, rules="unicode"))
        ascii_text = str(sp.regtable(*models, rules="ascii"))
        assert len(unicode_text.splitlines()) == len(ascii_text.splitlines())
        for line_u, line_a in zip(unicode_text.splitlines(), ascii_text.splitlines()):
            assert len(line_u) == len(line_a)
        assert "0.0932" in ascii_text and "0.2234" in ascii_text

    def test_unicode_mode_keeps_the_box_drawing_rules(self, models):
        assert "━" in str(sp.regtable(*models, rules="unicode"))

    def test_transposed_render_honours_the_mode(self, models):
        text = str(sp.regtable(*models, transpose=True, rules="ascii"))
        assert text.isascii()

    def test_auto_degrades_when_stdout_cannot_encode_the_rules(
        self, models, monkeypatch
    ):
        monkeypatch.setattr(
            sys,
            "stdout",
            io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict"),
        )
        text = str(sp.regtable(*models))
        assert text.isascii()
        _as_cp1252(text)

    def test_auto_keeps_unicode_on_a_utf8_stdout(self, models, monkeypatch):
        monkeypatch.setattr(
            sys,
            "stdout",
            io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict"),
        )
        assert "━" in str(sp.regtable(*models))

    def test_unknown_mode_is_rejected_at_construction(self, models):
        with pytest.raises(sp.MethodIncompatibility, match="rules must be"):
            sp.regtable(*models, rules="fancy")
