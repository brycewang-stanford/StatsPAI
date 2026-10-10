"""Citations for synthetic-control estimators, taken from ``paper.bib``."""

from __future__ import annotations

from ..core.results import CausalResult


def bib_citation(key: str, fallback: str) -> str:
    """Register the ``paper.bib`` entry ``key`` and return the key to cite.

    ``CausalResult.cite()`` looks its key up in the curated ``_CITATIONS``
    table and otherwise falls back to a substring match, which hands any
    key containing ``synth`` the 2010 synthetic-control paper. An estimator
    whose reference is already a verified ``paper.bib`` entry registers that
    entry verbatim under its bib key instead, so nothing is retyped. If the
    master bib cannot be read, ``fallback`` is returned and the result
    cites as it did before.
    """
    if key in CausalResult._CITATIONS:
        return key
    from ..smart.citations import bibtex

    try:
        entry = bibtex(key)
    except (KeyError, ValueError, FileNotFoundError):
        return fallback
    CausalResult._CITATIONS[key] = entry
    return key
