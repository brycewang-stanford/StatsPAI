"""Locate the master bibliography (``paper.bib``) at runtime.

``paper.bib`` is StatsPAI's single source of truth for verified references
(CLAUDE.md §10): ``sp.bibtex()``, the MCP ``bibtex`` tool, and the §10
cross-checks in ``sp.recommend_benchmark`` all read it. Two copies exist:

* the repository root ``paper.bib`` — the file that is edited and audited;
* ``statspai/paper.bib`` inside the package — a byte-identical copy that
  ships in the wheel so installed users resolve the same entries
  (``tools/bib_subset.py packaged --check`` keeps the two in sync).

Resolution prefers the repository copy when running from a source checkout
(so an edit is visible immediately under an editable install) and falls
back to the packaged copy otherwise. There is deliberately **no**
current-working-directory fallback: a user's own ``paper.bib`` must never be
mistaken for StatsPAI's verified bibliography.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

_PACKAGE_DIR = Path(__file__).resolve().parent


def master_bib_path() -> Path:
    """Return the path of the verified master ``paper.bib``.

    Raises
    ------
    FileNotFoundError
        If neither the source-checkout copy nor the packaged copy exists.
        This is loud on purpose: silently returning an empty bibliography
        would let a caller "resolve" a key to nothing and invent a citation.
    """
    candidates = []
    # Source checkout (src layout): <repo>/src/statspai -> <repo>/paper.bib.
    # Only trust the two-levels-up file when the layout really is "src/",
    # so an unrelated paper.bib that happens to sit two directories above an
    # installed package is never picked up.
    if _PACKAGE_DIR.parent.name == "src":
        candidates.append(_PACKAGE_DIR.parent.parent / "paper.bib")
    candidates.append(_PACKAGE_DIR / "paper.bib")  # packaged copy inside the wheel
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "StatsPAI's master bibliography (paper.bib) was not found at "
        + " or ".join(str(c) for c in candidates)
        + ". The wheel ships statspai/paper.bib; a broken install or a "
        "source tree without paper.bib cannot resolve citations (CLAUDE.md §10)."
    )


def read_master_bib(encoding: str = "utf-8") -> str:
    """Return the text of the master ``paper.bib`` (see :func:`master_bib_path`)."""
    return master_bib_path().read_text(encoding=encoding)


def master_bib_path_or_none() -> Optional[Path]:
    """Like :func:`master_bib_path` but return ``None`` instead of raising."""
    try:
        return master_bib_path()
    except FileNotFoundError:
        return None
