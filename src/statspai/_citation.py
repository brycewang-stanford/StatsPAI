"""Package-level citation helpers for StatsPAI.

Use :func:`citation` to get a BibTeX (or APA / plain) citation string for the
package itself.  ``sp.__citation__`` is a convenience attribute that holds the
default BibTeX entry as a plain ``str``.

For inline coefficient-level citations inside running text (e.g. rendering
``"β = 0.34** (0.12)"``), use :func:`statspai.cite` instead — that's a
different function with a different purpose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

__all__ = ["citation"]


_RELEASE_YEAR = "2026"

# Zenodo *concept* DOI — always resolves to the latest archived release.
# Update when Zenodo issues a new concept DOI (rare; usually only the version
# DOI changes).  The version-specific DOI for the current release is shipped
# in CITATION.cff under ``identifiers``.
_CONCEPT_DOI = "10.5281/zenodo.19933900"

_BIBTEX_TEMPLATE = (
    "@software{{wang_statspai_{year},\n"
    "  author       = {{Wang, Biaoyue}},\n"
    "  title        = {{StatsPAI: The Agent-Native Causal Inference \\&"
    " Econometrics Toolkit for Python}},\n"
    "  year         = {{{year}}},\n"
    "  version      = {{{version}}},\n"
    "  doi          = {{{doi}}},\n"
    "  url          = {{https://doi.org/{doi}}},\n"
    "  license      = {{MIT}},\n"
    "}}"
)

_APA_TEMPLATE = (
    "Wang, B. ({year}). StatsPAI: The Agent-Native Causal Inference & "
    "Econometrics Toolkit for Python (Version {version}) [Computer software]. "
    "Zenodo. https://doi.org/{doi}"
)

_PLAIN_TEMPLATE = (
    "Biaoyue Wang ({year}). StatsPAI: The Agent-Native Causal Inference & "
    "Econometrics Toolkit for Python, version {version}. "
    "https://doi.org/{doi}"
)


def _read_cff() -> Optional[str]:
    """Return CITATION.cff contents if reachable from an editable install."""
    here = Path(__file__).resolve()
    candidates = (
        here.parent.parent.parent / "CITATION.cff",  # repo root, editable install
        here.parent / "CITATION.cff",                # in-package copy, if shipped
    )
    for path in candidates:
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8")
        except OSError:
            continue
    return None


def citation(format: str = "bibtex") -> str:
    """Return a citation string for the StatsPAI package.

    Parameters
    ----------
    format : {"bibtex", "apa", "plain", "cff"}, default ``"bibtex"``
        - ``"bibtex"`` — BibTeX entry suitable for LaTeX bibliographies.
        - ``"apa"``    — APA-style human-readable string.
        - ``"plain"``  — Minimal plain-text string.
        - ``"cff"``    — Raw contents of the repository ``CITATION.cff`` file
          (only available in editable / source installs that ship the file).

    Returns
    -------
    str
        The citation string.

    Notes
    -----
    The JOSS paper for StatsPAI is currently under review.  Once accepted,
    this function will return the journal article citation as the preferred
    form; until then, please cite the software entry (and, if available, the
    versioned Zenodo DOI for the specific release you used).

    For formatting a single coefficient as inline text (e.g.
    ``"β = 0.34** (0.12)"``), use :func:`statspai.cite` instead.

    Examples
    --------
    >>> import statspai as sp
    >>> print(sp.citation())             # BibTeX (default)
    >>> print(sp.citation("apa"))        # APA
    >>> sp.__citation__                  # same as sp.citation("bibtex")
    """
    from . import __version__

    fmt = format.lower()
    kwargs = dict(year=_RELEASE_YEAR, version=__version__, doi=_CONCEPT_DOI)
    if fmt == "bibtex":
        return _BIBTEX_TEMPLATE.format(**kwargs)
    if fmt == "apa":
        return _APA_TEMPLATE.format(**kwargs)
    if fmt == "plain":
        return _PLAIN_TEMPLATE.format(**kwargs)
    if fmt == "cff":
        cff = _read_cff()
        if cff is None:
            raise FileNotFoundError(
                "CITATION.cff not found alongside the installed package; "
                "only available in editable / source installs."
            )
        return cff
    raise ValueError(
        f"format={format!r} invalid; choose from "
        "'bibtex', 'apa', 'plain', 'cff'."
    )
