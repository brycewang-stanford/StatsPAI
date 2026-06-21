"""
Tests for auto-CJK fallback registration.

Behavior contract (see ``statspai.plots.themes._register_cjk_fallback``):

1. The user's primary font family stays at ``font.family[0]`` — Latin glyphs
   are rendered with their original primary font, unchanged.
2. CJK font names are APPENDED to ``font.family`` (not prepended, not
   replacing) so matplotlib's per-glyph fallback (mpl 3.6+) walks them only
   for codepoints missing from the primary font.
3. ``axes.unicode_minus`` is NOT touched (Latin U+2212 stays correct).
4. ``font.sans-serif`` and ``font.serif`` are NOT touched (theme-managed
   lists are preserved as-is).
5. Rendering Chinese text after a plain ``import statspai as sp`` (no
   ``use_chinese()`` call) must produce zero "Glyph missing from font"
   warnings — provided a CJK font is installed.
6. User's explicit ``rcParams['font.family'] = ...`` after import must
   fully replace our list.
7. The ``STATSPAI_NO_AUTO_CJK=1`` env var must disable registration.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from statspai.plots import themes


def _has_cjk_font() -> bool:
    """Whether the test machine has any CJK font installed."""
    return bool(
        themes._get_cn_sans_fonts()
        or [
            f
            for f in themes._get_cn_serif_fonts()
            if f not in ("Times New Roman", "DejaVu Serif")
        ]
    )


def _glyph_warnings(records):
    return [
        str(r.message)
        for r in records
        if "Glyph" in str(r.message) or "missing from font" in str(r.message)
    ]


@pytest.fixture(autouse=True)
def _restore_rcparams():
    """Snapshot rcParams around every test so user-state mutations don't leak."""
    saved = mpl.rcParams.copy()
    yield
    mpl.rcParams.update(saved)


def test_primary_family_stays_first():
    """User's primary family stays at font.family[0]. Latin glyphs render
    with the primary font unchanged."""
    mpl.rcParams["font.family"] = ["sans-serif"]  # matplotlib default

    themes._register_cjk_fallback(force=True)

    family = list(mpl.rcParams["font.family"])
    assert (
        family[0] == "sans-serif"
    ), f"primary family must remain at index 0; got: {family}"


def test_import_does_not_change_unicode_minus():
    """We must NOT flip axes.unicode_minus globally; that degrades Latin plots."""
    mpl.rcParams["axes.unicode_minus"] = True  # matplotlib default
    themes._register_cjk_fallback(force=True)
    assert (
        mpl.rcParams["axes.unicode_minus"] is True
    ), "axes.unicode_minus must not be touched by auto-CJK registration"


def test_does_not_touch_family_specific_lists():
    """font.sans-serif and font.serif (theme-managed primary-family fallbacks)
    must be left alone by auto-registration."""
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    mpl.rcParams["font.serif"] = ["DejaVu Serif"]

    themes._register_cjk_fallback(force=True)

    assert list(mpl.rcParams["font.sans-serif"]) == ["DejaVu Sans"]
    assert list(mpl.rcParams["font.serif"]) == ["DejaVu Serif"]


@pytest.mark.skipif(not _has_cjk_font(), reason="no CJK font on this system")
def test_cjk_font_appended_to_family_list():
    """CJK font(s) go at the END of font.family so user's primary wins for
    Latin characters and only CJK falls through."""
    mpl.rcParams["font.family"] = ["sans-serif"]

    themes._register_cjk_fallback(force=True)
    info = themes._cjk_fallback_info
    assert not info["skipped"], f"unexpectedly skipped: {info['reason']}"
    assert info["appended"], "expected at least one CJK font to be appended"

    family = list(mpl.rcParams["font.family"])
    assert family[0] == "sans-serif", f"primary moved: {family}"
    for cjk_font in info["appended"]:
        assert cjk_font in family, f"CJK font {cjk_font!r} not in font.family: {family}"


@pytest.mark.skipif(not _has_cjk_font(), reason="no CJK font on this system")
def test_chinese_renders_without_warnings_after_register():
    """After registration, rendering a plot with Chinese labels must produce
    zero glyph-missing warnings."""
    themes._register_cjk_fallback(force=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title("中文标题：处理效应")
        ax.set_xlabel("年份")
        ax.set_ylabel("收入对数")
        ax.legend(["处理组"])
        fig.canvas.draw()  # force rendering so glyphs are resolved
        plt.close(fig)

    glyph_warns = _glyph_warnings(caught)
    assert (
        not glyph_warns
    ), f"unexpected glyph warnings after auto-CJK register:\n" + "\n".join(
        glyph_warns[:5]
    )


def test_user_override_wins():
    """If user sets font.family explicitly AFTER import, their value replaces
    ours fully — no leftover CJK entries from our injection."""
    themes._register_cjk_fallback(force=True)

    # User explicitly overrides — assignment must replace, not extend.
    mpl.rcParams["font.family"] = "Times New Roman"
    assert mpl.rcParams["font.family"] == ["Times New Roman"]


def test_idempotent_without_force():
    """Calling register twice without force must be a no-op the second time."""
    mpl.rcParams["font.family"] = ["sans-serif"]

    themes._register_cjk_fallback(force=True)
    after_first = list(mpl.rcParams["font.family"])

    themes._register_cjk_fallback(force=False)  # should be no-op
    after_second = list(mpl.rcParams["font.family"])

    assert after_first == after_second


def test_env_var_opt_out(monkeypatch):
    """STATSPAI_NO_AUTO_CJK=1 must skip registration entirely."""
    monkeypatch.setenv("STATSPAI_NO_AUTO_CJK", "1")
    mpl.rcParams["font.family"] = ["sans-serif"]

    info = themes._register_cjk_fallback(force=True)

    assert info["skipped"] is True
    assert "STATSPAI_NO_AUTO_CJK" in info["reason"]
    # font.family untouched
    assert list(mpl.rcParams["font.family"]) == ["sans-serif"]


def test_register_called_on_plots_import():
    """Importing statspai must have already triggered registration."""
    # statspai.plots is imported transitively when the test imports `themes`.
    # If the wiring is correct, _cjk_fallback_registered is already True.
    import statspai  # noqa: F401  (ensure top-level import path is exercised)

    assert themes._cjk_fallback_registered is True
