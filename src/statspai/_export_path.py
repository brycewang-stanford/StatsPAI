"""File-suffix check shared by the ``to_excel`` / ``to_word`` / ``to_latex`` exporters.

Every result class writes its own file, and none looked at the name it was
given: ``result.to_excel("t.docx")`` wrote xlsx bytes under a Word name, and
``to_excel("table")`` a file no spreadsheet program opens by name (2026-11-05
rOpenSci audit, G4.0). The rule, in one place for every exporter:

* a path without a suffix (or with one that is not a file format, such as
  ``"results.v2"``) gets the format's suffix appended -- Stata's
  ``putexcel set`` and ``outreg2`` do the same;
* a path whose suffix names another format (``.docx`` given to
  ``to_excel``) is refused with :class:`~statspai.exceptions.MethodIncompatibility`;
* anything that is not a ``str`` / ``os.PathLike`` (a buffer, ``None`` for
  "return the text") passes through untouched.
"""

from __future__ import annotations

import functools
import inspect
import os
from typing import Any, Callable, Dict, FrozenSet, Tuple, TypeVar, cast

__all__ = ["export_path", "checks_export_path"]

_F = TypeVar("_F", bound=Callable[..., Any])

#: format -> (accepted suffixes, suffix appended when none is given, label)
_FORMATS: Dict[str, Tuple[FrozenSet[str], str, str]] = {
    "excel": (frozenset({".xlsx", ".xlsm"}), ".xlsx", "an Excel workbook"),
    "word": (frozenset({".docx"}), ".docx", "a Word document"),
    "latex": (frozenset({".tex", ".ltx", ".txt"}), ".tex", "a LaTeX table"),
}

#: Suffixes that name a file format, so a mismatch is an error rather than
#: part of a file name.
_KNOWN_FORMATS = frozenset(
    {
        ".xlsx", ".xlsm", ".xls", ".ods", ".csv", ".tsv",
        ".docx", ".doc", ".odt", ".rtf", ".pdf",
        ".tex", ".ltx", ".txt", ".md", ".markdown", ".qmd", ".rmd",
        ".html", ".htm", ".json", ".xml", ".typ",
        ".png", ".jpg", ".jpeg", ".svg", ".pptx",
        ".parquet", ".feather", ".dta", ".pkl", ".pickle",
    }
)  # fmt: skip

_OTHER_METHOD = {"excel": "to_excel", "word": "to_word", "latex": "to_latex"}


def export_path(path: Any, fmt: str, *, method: str) -> Any:
    """``path`` checked against the format ``fmt`` (see the module docstring)."""
    if not isinstance(path, (str, os.PathLike)):
        return path
    accepted, default, label = _FORMATS[fmt]
    text = os.fspath(path)
    if not isinstance(text, str):  # a bytes path: leave it to the writer
        return path
    suffix = os.path.splitext(text)[1].lower()
    if suffix in accepted:
        return path
    if suffix in _KNOWN_FORMATS:
        from .exceptions import MethodIncompatibility

        fitting = next(
            (
                _OTHER_METHOD[f]
                for f, (sfx, _, _) in _FORMATS.items()
                if f != fmt and suffix in sfx
            ),
            None,
        )
        raise MethodIncompatibility(
            f"{method}: {text!r} ends in {suffix!r}, but {method} writes "
            f"{label} ({' / '.join(sorted(accepted))}).",
            recovery_hint=(
                f"Use a path ending in {default}"
                + (
                    f", or call .{fitting}() to write a {suffix} file."
                    if fitting
                    else "."
                )
            ),
            diagnostics={"path": text, "suffix": suffix, "format": fmt},
        )
    new = text + default
    return type(path)(new) if not isinstance(path, str) else new


def checks_export_path(fmt: str, *, param: str = "") -> Callable[[_F], _F]:
    """Decorator: run :func:`export_path` on an exporter's path argument.

    ``param`` names the path parameter; by default the first of ``path`` /
    ``filename`` in the signature. The wrapped method receives (and, when it
    returns its path, returns) the checked path.
    """

    def deco(fn: _F) -> _F:
        sig = inspect.signature(fn)
        names = list(sig.parameters)
        name = param or next((n for n in ("path", "filename") if n in names), "")
        if not name:
            raise TypeError(f"{fn.__qualname__} has no path parameter")
        index = names.index(name)
        positional = sig.parameters[name].kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        label = f".{fn.__name__}()"

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if positional and len(args) > index:
                args = (
                    *args[:index],
                    export_path(args[index], fmt, method=label),
                    *args[index + 1 :],
                )
            elif name in kwargs:
                kwargs[name] = export_path(kwargs[name], fmt, method=label)
            return fn(*args, **kwargs)

        return cast(_F, wrapper)

    return deco
