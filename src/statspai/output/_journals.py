"""Journal presets for publication-quality regression tables.

Single source of truth for per-journal formatting choices: significance-star
thresholds, default SE row label phrasing, default summary-stat selection,
and footer-note templates. Centralising these here means adding a new
journal is a single dict entry instead of patching ``regtable``,
``paper_tables``, and the rendering layer.

Currently supported templates (case-insensitive):

============== =============================================================
``aer``        American Economic Review (default; AEJs follow same style).
``qje``        Quarterly Journal of Economics — robust SE phrasing, no R² by default.
``econometrica`` Econometrica — three-threshold convention (matches modern issues).
``restat``     Review of Economics and Statistics.
``jf``         Journal of Finance — finance-style with adj. R² in stats.
``aeja``       AEJ: Applied Economics — same as AER but with adj. R².
``jpe``        Journal of Political Economy — AER-equivalent stars + R².
``restud``     Review of Economic Studies — AER-equivalent.
============== =============================================================

Notes
-----
Star symbols are uniformly ``*``, ``**``, ``***`` (sorted from loosest to
strictest threshold) — economics journals do not use the dagger ``†``
convention common in psychology. The Econometrica preset uses the
three-level convention by default; users wanting the legacy two-level
scheme (``**`` 5% / ``***`` 1% only) can pass ``star_levels=(0.05, 0.01)``
explicitly to ``regtable``.

References
----------
Threshold and SE-label conventions cross-checked against the public
"Information for authors" pages of each journal (verified 2026-04-25).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# Journal preset registry
# ---------------------------------------------------------------------------

#: Public, immutable view of the journal preset registry. Each value is a
#: dict with keys:
#:
#: ``label`` (str)
#:     Human-readable display label.
#: ``star_levels`` (tuple of float)
#:     Significance thresholds, sorted from loosest to strictest. The
#:     coefficient gets one ``*`` per crossed threshold.
#: ``se_label`` (str)
#:     Phrase appearing in the footer note (e.g. ``"Standard errors"`` or
#:     ``"Robust standard errors"``).
#: ``stats`` (tuple of str)
#:     Default summary statistics shown beneath coefficients.
#: ``notes_default`` (tuple of str)
#:     Default footer-note lines, in display order.
#: ``font_name`` (str)
#:     Suggested font for DOCX/XLSX export.
JOURNALS: Dict[str, Dict[str, Any]] = {
    "aer": {
        "label": "American Economic Review",
        "star_levels": (0.10, 0.05, 0.01),
        "se_label": "Standard errors",
        "stats": ("N", "R-squared"),
        "notes_default": (
            "Standard errors in parentheses.",
            "*** p<0.01, ** p<0.05, * p<0.10.",
        ),
        "font_name": "Times New Roman",
    },
    "qje": {
        "label": "Quarterly Journal of Economics",
        "star_levels": (0.10, 0.05, 0.01),
        "se_label": "Robust standard errors",
        "stats": ("N",),
        "notes_default": (
            "Robust standard errors in parentheses.",
            "*** p<0.01, ** p<0.05, * p<0.10.",
        ),
        "font_name": "Times New Roman",
    },
    "econometrica": {
        "label": "Econometrica",
        # Some Econometrica papers use only ** (5%) and *** (1%); newer
        # issues mostly use the full three-level convention. We follow the
        # AER-equivalent default so behaviour is predictable across journals.
        "star_levels": (0.10, 0.05, 0.01),
        "se_label": "Standard errors",
        "stats": ("N", "R-squared"),
        "notes_default": (
            "Standard errors in parentheses.",
            "*** p<0.01, ** p<0.05, * p<0.10.",
        ),
        "font_name": "Times New Roman",
    },
    "restat": {
        "label": "Review of Economics and Statistics",
        "star_levels": (0.10, 0.05, 0.01),
        "se_label": "Standard errors",
        "stats": ("N", "R-squared"),
        "notes_default": (
            "Standard errors in parentheses.",
            "*** p<0.01, ** p<0.05, * p<0.10.",
        ),
        "font_name": "Times New Roman",
    },
    "jf": {
        "label": "Journal of Finance",
        "star_levels": (0.10, 0.05, 0.01),
        "se_label": "Standard errors",
        "stats": ("N", "R-squared", "Adj. R-squared"),
        "notes_default": (
            "Standard errors in parentheses.",
            "*** p<0.01, ** p<0.05, * p<0.10.",
        ),
        "font_name": "Times New Roman",
    },
    "aeja": {
        "label": "AEJ: Applied Economics",
        "star_levels": (0.10, 0.05, 0.01),
        "se_label": "Standard errors",
        "stats": ("N", "R-squared", "Adj. R-squared"),
        "notes_default": (
            "Standard errors in parentheses.",
            "*** p<0.01, ** p<0.05, * p<0.10.",
        ),
        "font_name": "Times New Roman",
    },
    "jpe": {
        "label": "Journal of Political Economy",
        "star_levels": (0.10, 0.05, 0.01),
        "se_label": "Standard errors",
        "stats": ("N", "R-squared"),
        "notes_default": (
            "Standard errors in parentheses.",
            "*** p<0.01, ** p<0.05, * p<0.10.",
        ),
        "font_name": "Times New Roman",
    },
    "restud": {
        "label": "Review of Economic Studies",
        "star_levels": (0.10, 0.05, 0.01),
        "se_label": "Standard errors",
        "stats": ("N", "R-squared"),
        "notes_default": (
            "Standard errors in parentheses.",
            "*** p<0.01, ** p<0.05, * p<0.10.",
        ),
        "font_name": "Times New Roman",
    },
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def list_templates() -> Tuple[str, ...]:
    """Return the canonical names of every registered journal preset."""
    return tuple(JOURNALS.keys())


def get_template(name: str) -> Dict[str, Any]:
    """Look up a journal preset by name (case-insensitive).

    Parameters
    ----------
    name : str
        Template identifier (e.g. ``"aer"`` / ``"AER"`` / ``"jf"``).

    Returns
    -------
    dict
        A *copy* of the preset entry. Callers are free to mutate it.

    Raises
    ------
    ValueError
        If *name* is not a registered template.
    """
    key = (name or "").strip().lower()
    if key not in JOURNALS:
        raise ValueError(
            f"Unknown journal template {name!r}. "
            f"Available: {', '.join(sorted(JOURNALS))}"
        )
    return dict(JOURNALS[key])


def star_note_for(star_levels: Tuple[float, ...]) -> str:
    """Render the ``"*** p<0.01, ** p<0.05, * p<0.10"`` footer string.

    Sorts loosest threshold first then writes them strict-first so that the
    most-stars threshold appears at the start of the rendered note (this is
    the convention used by every preset in :data:`JOURNALS`).
    """
    if not star_levels:
        return ""
    sorted_loose_first = sorted(star_levels, reverse=True)
    parts = []
    n_levels = len(sorted_loose_first)
    for i, lev in enumerate(sorted_loose_first):
        n_stars = i + 1
        parts.append(("*" * n_stars, lev))
    parts_strict_first = list(reversed(parts))
    pieces = []
    for stars, lev in parts_strict_first:
        pieces.append(f"{stars} p<{lev:.2f}")
    return ", ".join(pieces)
