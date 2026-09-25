"""ASCII normalisation of source files, shared by every hash gate.

The JSS submission archive must carry ASCII source files, so the packager
(``Paper-JSS/replication/scripts/jss_submission_package.py``) transliterates
``.py`` / ``.R`` / ``.do`` / ``.sh`` / ``.toml`` members that contain
non-ASCII characters (an em dash in a comment, a Greek letter in a
docstring). Two gates hash those same files: the Tier A fixture lock
(``scripts/tier_a_fixture_lock.py``) and the Track A implementation trace
(``scripts/trace_parity_provenance.py``, checked by
``tests/test_parity_implementation_provenance.py``). Hashing raw bytes made
both gates fail on the extracted archive although no code had changed.

Every gate therefore hashes :func:`normalized_source_bytes`: the file's
bytes with line endings folded to LF and, for a source file, the same ASCII
transliteration the archive applies. On an ASCII file both steps are the
identity, so its hash is unchanged; on a transliterated archive member the
hash equals that of the original. The packager imports
:func:`ascii_source_text` from here, so the archive and the gates cannot
drift apart.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

__all__ = [
    "ASCII_SOURCE_SUFFIXES",
    "TRANSLITERATION",
    "ascii_source_text",
    "normalized_source_bytes",
]

#: Suffixes the archive rewrites to ASCII.
ASCII_SOURCE_SUFFIXES = frozenset({".py", ".R", ".do", ".sh", ".toml"})

TRANSLITERATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "--",
        "\u2015": "--",
        "\u2212": "-",
        "\u2026": "...",
        "\u00d7": "x",
        "\u00b7": "*",
        "\u2248": "~",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2260": "!=",
        "\u2261": "==",
        "\u221e": "inf",
        "\u2208": "in",
        "\u2190": "<-",
        "\u2192": "->",
        "\u2194": "<->",
        "\u21d2": "=>",
        "\u221a": "sqrt",
        "\u2211": "sum",
        "\u222b": "integral",
        "\u2202": "partial",
        "\u22a5": "perp",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u00a7": "Section ",
        "\u00b1": "+/-",
        "\u2022": "*",
        "\u26a0": "WARNING",
        "\u2705": "OK",
        "\u2713": "OK",
        "\ufe0f": "",
        "\u2500": "-",
        "\u2501": "-",
        "\u2550": "=",
        "\u2502": "|",
        "\u2503": "|",
        "\u250c": "+",
        "\u2510": "+",
        "\u2514": "+",
        "\u2518": "+",
        "\u251c": "+",
        "\u2524": "+",
        "\u253c": "+",
        "\u2554": "+",
        "\u2557": "+",
        "\u255a": "+",
        "\u255d": "+",
        "\u2551": "|",
        "\u03b1": "alpha",
        "\u03b2": "beta",
        "\u03b3": "gamma",
        "\u03b4": "delta",
        "\u03b5": "epsilon",
        "\u03b7": "eta",
        "\u03b8": "theta",
        "\u03ba": "kappa",
        "\u03bb": "lambda",
        "\u03bc": "mu",
        "\u03bd": "nu",
        "\u03c0": "pi",
        "\u03c1": "rho",
        "\u03c3": "sigma",
        "\u03c4": "tau",
        "\u03c6": "phi",
        "\u03c7": "chi",
        "\u03c8": "psi",
        "\u03c9": "omega",
        "\u0393": "Gamma",
        "\u0394": "Delta",
        "\u03a3": "Sigma",
        "\u03a6": "Phi",
        "\u03a8": "Psi",
        "\u03a9": "Omega",
        "\u2080": "0",
        "\u2081": "1",
        "\u2082": "2",
        "\u2083": "3",
        "\u2084": "4",
        "\u2085": "5",
        "\u2086": "6",
        "\u2087": "7",
        "\u2088": "8",
        "\u2089": "9",
        "\u00b9": "1",
        "\u00b2": "2",
        "\u00b3": "3",
        "\u2070": "0",
        "\u2074": "4",
        "\u2075": "5",
        "\u2076": "6",
        "\u2077": "7",
        "\u2078": "8",
        "\u2079": "9",
        "\u207b": "-",
    }
)


def ascii_source_text(text: str) -> str:
    """Return ``text`` as the ASCII the JSS archive ships."""
    text = text.replace("\u5f85\u6838\u9a8c", "pending verification")
    text = text.translate(TRANSLITERATION)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def normalized_source_bytes(path: Path) -> bytes:
    """The bytes a hash gate should hash for ``path``.

    Line endings are folded to LF (Windows checkouts under
    ``core.autocrlf``), and a non-ASCII source file is transliterated as the
    archive transliterates it.
    """
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if path.suffix not in ASCII_SOURCE_SUFFIXES:
        return data
    try:
        data.decode("ascii")
        return data
    except UnicodeDecodeError:
        return ascii_source_text(data.decode("utf-8")).encode("ascii")
