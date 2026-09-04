"""Package-level citation helpers for StatsPAI.

Use :func:`citation` to get a BibTeX (or APA / plain) citation string for
StatsPAI — by default the peer-reviewed JOSS article (Wang & Rozelle, 2026,
DOI 10.21105/joss.10604); pass ``which="software"`` for the versioned Zenodo
software entry.  ``sp.__citation__`` is a convenience attribute that holds the
default BibTeX entry as a plain ``str``.

For inline coefficient-level citations inside running text (e.g. rendering
``"β = 0.34** (0.12)"``), use :func:`statspai.cite` instead — that's a
different function with a different purpose.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Optional

__all__ = ["citation"]


_RELEASE_YEAR = "2026"

# Zenodo *concept* DOI — always resolves to the latest archived release.
# Update when Zenodo issues a new concept DOI (rare; usually only the version
# DOI changes).  The version-specific DOI for the current release is shipped
# in CITATION.cff under ``identifiers``.
_CONCEPT_DOI = "10.5281/zenodo.19933900"

# JOSS article — the preferred citation.  Published 2026-09-03; volume /
# issue / page verified against the Crossref deposit in
# openjournals/joss-papers (joss.10604/10.21105.joss.10604.crossref.xml).
_ARTICLE_DOI = "10.21105/joss.10604"
_ARTICLE_YEAR = "2026"
_ARTICLE_VOLUME = "11"
_ARTICLE_ISSUE = "125"
_ARTICLE_PAGE = "10604"

_ARTICLE_BIBTEX_TEMPLATE = (
    "@article{{wang{year}statspaijoss,\n"
    "  author       = {{Wang, Biaoyue and Rozelle, Scott}},\n"
    "  title        = {{StatsPAI: A Unified, Agent-Native Python Toolkit"
    " for Causal Inference and Applied Econometrics}},\n"
    "  journal      = {{Journal of Open Source Software}},\n"
    "  year         = {{{year}}},\n"
    "  volume       = {{{volume}}},\n"
    "  number       = {{{issue}}},\n"
    "  pages        = {{{page}}},\n"
    "  doi          = {{{doi}}},\n"
    "  url          = {{https://doi.org/{doi}}},\n"
    "}}"
)

_ARTICLE_APA_TEMPLATE = (
    "Wang, B., & Rozelle, S. ({year}). StatsPAI: A Unified, Agent-Native "
    "Python Toolkit for Causal Inference and Applied Econometrics. "
    "Journal of Open Source Software, {volume}({issue}), {page}. "
    "https://doi.org/{doi}"
)

_ARTICLE_PLAIN_TEMPLATE = (
    "Biaoyue Wang and Scott Rozelle ({year}). StatsPAI: A Unified, "
    "Agent-Native Python Toolkit for Causal Inference and Applied "
    "Econometrics. Journal of Open Source Software {volume}({issue}): {page}. "
    "https://doi.org/{doi}"
)

_BIBTEX_TEMPLATE = (
    "@software{{wang{year}statspai,\n"
    "  author       = {{Wang, Biaoyue and Rozelle, Scott}},\n"
    "  title        = {{StatsPAI: A Unified, Agent-Native Python Toolkit"
    " for Causal Inference and Applied Econometrics}},\n"
    "  year         = {{{year}}},\n"
    "  version      = {{{version}}},\n"
    "  doi          = {{{doi}}},\n"
    "  url          = {{https://doi.org/{doi}}},\n"
    "  license      = {{MIT}},\n"
    "}}"
)

_APA_TEMPLATE = (
    "Wang, B., & Rozelle, S. ({year}). StatsPAI: A Unified, Agent-Native "
    "Python Toolkit for Causal Inference and Applied Econometrics "
    "(Version {version}) [Computer software]. "
    "Zenodo. https://doi.org/{doi}"
)

_PLAIN_TEMPLATE = (
    "Biaoyue Wang and Scott Rozelle ({year}). StatsPAI: A Unified, "
    "Agent-Native Python Toolkit for Causal Inference and Applied "
    "Econometrics, version {version}. "
    "https://doi.org/{doi}"
)


def _read_cff() -> Optional[str]:
    """Return CITATION.cff contents from package data or source checkout."""
    try:
        ref = resources.files("statspai").joinpath("CITATION.cff")
        if ref.is_file():
            return ref.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        pass

    here = Path(__file__).resolve()
    candidates = (
        here.parent.parent.parent / "CITATION.cff",  # repo root, editable install
    )
    for path in candidates:
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8")
        except OSError:
            continue
    return None


def citation(format: str = "bibtex", which: str = "paper") -> str:
    """Return a citation string for StatsPAI.

    Parameters
    ----------
    format : {"bibtex", "apa", "plain", "cff"}, default ``"bibtex"``
        - ``"bibtex"`` — BibTeX entry suitable for LaTeX bibliographies.
        - ``"apa"``    — APA-style human-readable string.
        - ``"plain"``  — Minimal plain-text string.
        - ``"cff"``    — Raw contents of the repository ``CITATION.cff`` file
          (ships with the package; ``which`` is ignored).
    which : {"paper", "software", "both"}, default ``"paper"``
        - ``"paper"``    — the peer-reviewed JOSS article
          (Wang & Rozelle, 2026, *J. Open Source Softw.* 11(125), 10604).
          This is the preferred citation.
        - ``"software"`` — the versioned software entry (Zenodo concept DOI
          plus the installed ``__version__``); use it when the exact release
          matters, e.g. in a replication package.
        - ``"both"``     — the two entries separated by a blank line.

    Returns
    -------
    str
        The citation string.

    Notes
    -----
    The JOSS paper was published on 2026-09-03 (DOI ``10.21105/joss.10604``)
    and is the preferred form; the software entry is the same one exposed as
    ``CITATION.cff``'s top-level metadata. ``sp.__citation__`` holds
    ``citation("bibtex")``, i.e. the article entry.

    For formatting a single coefficient as inline text (e.g.
    ``"β = 0.34** (0.12)"``), use :func:`statspai.cite` instead.

    Examples
    --------
    >>> import statspai as sp
    >>> print(sp.citation())                    # JOSS article, BibTeX
    >>> print(sp.citation("apa"))               # JOSS article, APA
    >>> print(sp.citation(which="software"))    # versioned software entry
    >>> print(sp.citation("plain", which="both"))
    """
    from . import __version__

    fmt = format.lower()
    if fmt == "cff":
        cff = _read_cff()
        if cff is None:
            raise FileNotFoundError(
                "CITATION.cff not found alongside the installed package; "
                "only available in editable / source installs."
            )
        return cff
    if fmt not in ("bibtex", "apa", "plain"):
        raise ValueError(
            f"format={format!r} invalid; choose from "
            "'bibtex', 'apa', 'plain', 'cff'."
        )

    kind = which.lower()
    if kind not in ("paper", "software", "both"):
        raise ValueError(
            f"which={which!r} invalid; choose from 'paper', 'software', 'both'."
        )

    paper_tpl = {
        "bibtex": _ARTICLE_BIBTEX_TEMPLATE,
        "apa": _ARTICLE_APA_TEMPLATE,
        "plain": _ARTICLE_PLAIN_TEMPLATE,
    }[fmt]
    software_tpl = {
        "bibtex": _BIBTEX_TEMPLATE,
        "apa": _APA_TEMPLATE,
        "plain": _PLAIN_TEMPLATE,
    }[fmt]

    paper = paper_tpl.format(
        year=_ARTICLE_YEAR,
        volume=_ARTICLE_VOLUME,
        issue=_ARTICLE_ISSUE,
        page=_ARTICLE_PAGE,
        doi=_ARTICLE_DOI,
    )
    software = software_tpl.format(
        year=_RELEASE_YEAR, version=__version__, doi=_CONCEPT_DOI
    )
    if kind == "paper":
        return paper
    if kind == "software":
        return software
    return paper + "\n\n" + software
